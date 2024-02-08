import argparse
import collections
import copy
import csv
import glob
import json
import os
import random
import re
import sys
from typing import List
import uuid

import json5
import jsonpath_ng
import jsonpath_ng.ext
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    Pipeline,
    pipeline,
)

import utils


# Uses code from https://github.com/dpfried/incoder/blob/main/example_usage.py

SPLIT_TOKEN = "<|insert|>"
EOM = "<|endofmask|>"
BOS = "<|endoftext|>"
FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
FIM_INDICATOR = "<FILL_HERE>"
DISALLOWED_CHARS = '"{}\\:'


def get_all_refs(obj, ref, prefix=jsonpath_ng.Root()):
    # Skip anything not a dictionary
    if not isinstance(obj, dict):
        return

    if obj.get("$ref") == ref:
        yield prefix
        return

    # Search any top-level definitions
    def_keys = ["definitions", "$defs"]
    for def_key in def_keys:
        if isinstance(prefix, jsonpath_ng.Root) and def_key in obj:
            for k, v in obj[def_key].items():
                yield from get_all_refs(
                    v,
                    ref,
                    jsonpath_ng.Child(
                        prefix,
                        jsonpath_ng.Child(
                            jsonpath_ng.Fields(def_key), jsonpath_ng.Fields(k)
                        ),
                    ),
                )

    prop_keys = ["properties", "patternProperties", "additionalProperties"]
    if obj.get("type") == "object":
        for prop_key in prop_keys:
            prop_obj = obj.get(prop_key, {})
            # additionalProperties can be a Boolean, so we check
            if not isinstance(prop_obj, dict):
                continue

            for k, v in prop_obj.items():
                yield from get_all_refs(
                    v,
                    ref,
                    jsonpath_ng.Child(
                        prefix,
                        jsonpath_ng.Child(
                            jsonpath_ng.Fields(prop_key), jsonpath_ng.Fields(k)
                        ),
                    ),
                )

    for k in ("allOf", "anyOf", "oneOf"):
        if k in obj:
            for i, v in enumerate(obj[k]):
                yield from get_all_refs(
                    v,
                    ref,
                    jsonpath_ng.Child(
                        prefix,
                        jsonpath_ng.Child(jsonpath_ng.Fields(k), jsonpath_ng.Index(i)),
                    ),
                )


def rename_key(old_name, new_name, reorder=False):
    # Strip quotes for special characters in paths
    #
    # Note that this will probably break if there is a single
    # quote in the path, but so far we have no examples of this.
    old_name = old_name.strip("'")

    def rename_fn(_data_field, data, field):
        if reorder:
            # Shuffle the items since the order shouldn't matter anyway.
            # This also helps make sure a diverse set of possible other
            # definitions are seen when we have to truncate for length.
            random_items = random.sample(list(data[field].items()), len(data[field]))

            # Build an OrderedDict to ensure the definition we're naming is first
            # This avoids it being truncated when the block of definitions is large
            data[field] = collections.OrderedDict(random_items)
            data[field][new_name] = data[field].pop(old_name)
            data[field].move_to_end(new_name, last=False)
        else:
            new_field = collections.OrderedDict()
            for k, v in data[field].items():
                if k == old_name:
                    new_field[new_name] = v
                else:
                    new_field[k] = v
            data[field] = new_field

    return rename_fn


def get_defn_paths(obj):
    def_keys = ["definitions", "$defs"]
    for def_key in def_keys:
        if def_key in obj:
            for k in obj[def_key]:
                yield jsonpath_ng.Child(
                    jsonpath_ng.Child(jsonpath_ng.Root(), jsonpath_ng.Fields(def_key)),
                    jsonpath_ng.Fields(k),
                )


def make_sentinel(i):
    # signals (1) a location to insert an infill and
    #         (2) the start of the infill generation
    return f"<|mask:{i}|>"


def generate(
    model,
    tokenizer,
    device,
    input: str,
    max_to_generate: int = 128,
    temperature: float = 0.2,
):
    """
    Do standard left-to-right completion of the
    prefix `input` by sampling from the model
    """
    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        sys.stderr.write(
            "warning: max_length {} is greater than the context window {}\n".format(
                max_length, 2048
            )
        )
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.95,
            temperature=temperature,
            max_length=max_length,
        )
    # pass clean_up_tokenization_spaces=False to avoid removing
    # spaces before punctuation, e.g. "from ." -> "from."
    detok_hypo_str = tokenizer.decode(
        output.flatten(), clean_up_tokenization_spaces=False
    )
    if detok_hypo_str.startswith(BOS):
        detok_hypo_str = detok_hypo_str[len(BOS) :]
    return detok_hypo_str


def infill(
    model,
    tokenizer,
    device,
    parts: List[str],
    max_to_generate: int = 128,
    temperature: float = 0.2,
    extra_sentinel: bool = True,
    max_retries: int = 1,
):
    """
    Generate infills to complete a partial document, e.g.
    [A C E] -> [A B C D E], where B and D are infills that have been generated.

    parts: List[str]. list of parts of the document. One string will be
            inserted in between each element, i.e. infilling N-1 locations for a list
            of length N.
    max_to_generate: int. maximum number of tokens to generate. Keep in mind
            that the model context size is 2048.
    temperature: float. temperature parameter for sampling.
    extra_sentinel: bool. we recommend setting this to True, as it makes it
            easier for the model to end generated infills. See the footnote in
            section 2.2 of our paper for details.
    max_retries: int. if > 1, use rejection sampling to keep sampling infills until
            all infills sample a completion token.

    returns a dictionary containing the following:
        text:  str, the completed document (with infills inserted)
        parts:  List[str], length N. Same as passed to the method
        infills:  List[str], length N-1. The list of infills generated
        retries_attempted:  number of retries used (if max_retries > 1)
    """
    assert isinstance(parts, list)
    retries_attempted = 0
    done = False

    while (not done) and (retries_attempted < max_retries):
        retries_attempted += 1

        ## (1) build the prompt
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)

        infills = []
        complete = []

        done = True

        ## (2) generate infills
        for sentinel_ix, part in enumerate(parts[:-1]):
            complete.append(part)
            prompt += make_sentinel(sentinel_ix)
            # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
            completion = generate(
                model, tokenizer, device, prompt, max_to_generate, temperature
            )
            completion = completion[len(prompt) :]
            if EOM not in completion:
                completion += EOM
                done = False
            completion = completion[: completion.index(EOM) + len(EOM)]
            infilled = completion[: -len(EOM)]
            infills.append(infilled)
            complete.append(infilled)
            prompt += completion
        complete.append(parts[-1])
        text = "".join(complete)

    return {
        "text": text,  # str, the completed document (with infills inserted)
        "parts": parts,  # List[str], length N. Same as passed to the method
        "infills": infills,  # List[str], length N-1. The list of infills generated
        "retries_attempted": retries_attempted,  # number of retries used
    }


def path_to_ref(path):
    return re.sub("^\$", "#", str(path)).replace(".", "/")


def replace_references(obj, old_path, new_path):
    if isinstance(obj, dict):
        obj = copy.deepcopy(obj)
        if "$ref" in obj and obj["$ref"] == path_to_ref(old_path):
            obj["$ref"] = path_to_ref(new_path)
        return collections.OrderedDict(
            (key, replace_references(value, old_path, new_path))
            for key, value in obj.items()
        )
    elif isinstance(obj, list):
        return [replace_references(item, old_path, new_path) for item in obj]
    else:
        return obj


def get_defn_template(schema, defn_path, token):
    # Get all references to this definition
    refs = get_all_refs(schema, path_to_ref(defn_path))

    # Build properties from the uses of the definition
    props = {}
    for r in refs:
        # Skip cases where the definition is used at the root
        if isinstance(r, jsonpath_ng.Root):
            continue

        key = str(r.right).split(".")[-1]
        if key not in props:
            # Make a copy and remove the reference
            obj = copy.copy(r.find(schema)[0].value)
            obj.pop("$ref")

            props[key] = obj

    return json.dumps(
        {
            "definitions": {token: defn_path.find(schema)[0].value},
            "properties": props,
        },
        indent=4,
    )[:514]


def generate_defn_name(schema, defn_path, model, tokenizer, mask_token, device):
    # We truncate the serialized JSON below to fit the model size
    defn_str = get_defn_template(schema, defn_path, SPLIT_TOKEN)[:514]

    if getattr(model, "task", None) == "fill-mask":
        return model(defn_str.replace(SPLIT_TOKEN, mask_token))[0]["token_str"]
    else:
        out = infill(model, tokenizer, device, defn_str.split(SPLIT_TOKEN))
        new_defn_name = out["infills"][0]

        # If a quote appears, keep anything before the quote
        if '"' in new_defn_name:
            quote_index = new_defn_name.index('"')
            new_defn_name = new_defn_name[:quote_index]

        return new_defn_name.strip()


def infill_defn_name(schema, defn_path, model, tokenizer, suppress_tokens, device):
    defn_str = get_defn_template(schema, defn_path, FIM_INDICATOR)
    # See https://huggingface.co/spaces/bigcode/bigcode-playground/blob/main/app.py
    prefix, suffix = defn_str.split(FIM_INDICATOR)
    defn_str = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

    # Encode the input and generate a definition name
    inputs = tokenizer(defn_str, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            do_sample=True,
            num_beams=3,
            top_p=0.9,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
        ),
        suppress_tokens=suppress_tokens,
    )
    generated_code = tokenizer.decode(
        output.flatten(), skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return utils.strip_generated_code(
        generated_code[generated_code.find(suffix) + len(suffix) :]
    )


def process_file(
    infile, outfile, strip_existing, model, tokenizer, suppress_tokens, device, args
):
    with open(infile, "r", encoding="utf-8") as f:
        json_str = f.read()
        obj = json5.loads(json_str)

    paths = list(get_defn_paths(obj))

    # Strip existing descriptions if requested
    if strip_existing:
        obj = utils.strip_meta(obj, utils.get_all_paths(obj))

    # Erase the existing definition names before continuing
    temp_names = set()
    if not args.keep_existing:
        orig_mapping = {}
        for i, defn_path in enumerate(paths):
            # Rename the definition in the object
            temp_name = uuid.uuid4().hex
            temp_names.add(temp_name)
            obj = defn_path.left.update_or_create(
                obj, rename_key(str(defn_path.right), temp_name)
            )

            # Replace the old path with the new path
            paths[i] = jsonpath_ng.Child(defn_path.left, jsonpath_ng.Fields(temp_name))
            obj = replace_references(obj, defn_path, paths[i])

            # Keep a mapping so we know the original definition name
            orig_mapping[paths[i]] = defn_path
    else:
        orig_mapping = {path: path for path in paths}

    defn_names = {}
    new_names = set()
    final_mapping = {}
    for defn_path in tqdm(paths, desc=os.path.basename(infile), leave=False):
        if args.model_name.startswith("bigcode/"):
            defn_name = infill_defn_name(
                obj, defn_path, model, tokenizer, suppress_tokens, device
            )
        else:
            if args.model_name.startswith("hf-internal-testing/"):
                mask_token = "[MASK]"
            else:
                mask_token = "<mask>"
            defn_name = generate_defn_name(
                obj, defn_path, model, tokenizer, mask_token, device
            )

        # If we somehow generated one of the random UUIDs, don't use it
        if defn_name in temp_names:
            defn_name = "defn"

        # Add a numerical suffix if needed
        if defn_name in new_names:
            defn_suffix = 2
            while (defn_name + str(defn_suffix)) in new_names:
                defn_suffix += 1
            defn_name += str(defn_suffix)
        new_names.add(defn_name)

        # Store this definition name to update later
        defn_names[defn_path] = defn_name
        final_mapping[orig_mapping[defn_path]] = jsonpath_ng.Child(
            defn_path.left, jsonpath_ng.Fields(defn_name)
        )

    # Iterate through all the collected definitions and update the object
    for defn_path, defn_name in defn_names.items():
        obj = defn_path.left.update_or_create(
            copy.deepcopy(obj), rename_key(str(defn_path.right), defn_name)
        )
        new_path = jsonpath_ng.Child(defn_path.left, jsonpath_ng.Fields(defn_name))
        obj = replace_references(obj, defn_path, new_path)

    # Output the mapping between old and new definitions
    if args.output_mapping:
        writer = csv.writer(sys.stderr)
        for orig, final in final_mapping.items():
            writer.writerow([orig, final])

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=utils.InputOutputType(True), default="/dev/stdin"
    )
    parser.add_argument(
        "-o", "--output", type=utils.InputOutputType(False), default="/dev/stdout"
    )
    parser.add_argument("-m", "--model-name", default="neulab/codebert-javascript")
    parser.add_argument("-c", "--cpu", default=False, action="store_true")
    parser.add_argument("-4", "--load-in-4bit", default=False, action="store_true")
    parser.add_argument("-8", "--load-in-8bit", default=False, action="store_true")
    parser.add_argument("-k", "--keep-existing", default=False, action="store_true")
    parser.add_argument("--output-mapping", default=False, action="store_true")
    parser.add_argument("--better-transformer", default=False, action="store_true")
    parser.add_argument("--skip-existing", default=False, action="store_true")
    parser.add_argument(
        "--no-strip-existing", dest="strip_existing", default=True, action="store_false"
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Load model
    sys.stderr.write("Loading model…\n")

    # Construct the quantization configuration
    qkwargs = {}
    if args.load_in_4bit:
        qkwargs["load_in_4bit"] = True
    if args.load_in_8bit:
        qkwargs["load_in_8bit"] = True
    qconfig = BitsAndBytesConfig(**qkwargs)

    if args.model_name.startswith("facebook/incoder-"):
        # Add model-specific parameters
        kwargs = {"quantization_config": qconfig}

        if args.model_name == "facebook/incoder-6B":
            kwargs["low_cpu_mem_usage"] = True
            if not args.cpu:
                kwargs["revision"] = "float16"
                kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True, device_map="auto", **kwargs
        )

        # Load tokenizer
        sys.stderr.write("Loading tokenizer…\n")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, device_map="auto"
        )
        tokenizer.pad_token = "<pad>"
        tokenizer.padding_side = "left"
    elif args.model_name.startswith("bigcode/"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=qconfig,
        )

        # Load tokenizer
        sys.stderr.write("Loading tokenizer…\n")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, device_map="auto"
        )
    else:
        if args.load_in_4bit or args.load_in_8bit:
            raise argparse.ArgumentTypeError(
                "quantization not supported for this model"
            )

        model = pipeline("fill-mask", model=args.model_name, device=device)
        tokenizer = None

    # If not using a quantized model, set to the correct device
    if (
        not args.load_in_4bit
        and not args.load_in_8bit
        and not isinstance(model, Pipeline)
    ):
        model = model.to(device)

    # Set the pad token if unspecified
    if tokenizer and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get tokens to suppress
    if tokenizer:
        suppress_tokens = [
            v
            for (k, v) in tokenizer.vocab.items()
            if any(c in DISALLOWED_CHARS for c in k) and v != tokenizer.all_special_ids
        ]
    else:
        suppress_tokens = []

    # Convert to BetterTransformer
    if args.better_transformer:
        model = model.to_bettertransformer()

    sys.stderr.write("Generating definition names…\n")
    if os.path.isfile(args.input):
        if os.path.isdir(args.output):
            args.output = os.path.join(args.output, os.path.basename(args.input))
        process_file(
            args.input,
            args.output,
            args.strip_existing,
            model,
            tokenizer,
            suppress_tokens,
            device,
            args,
        )
    elif os.path.isdir(args.input):
        # Generate the output directory if needed
        os.makedirs(args.output, exist_ok=True)

        for infile in tqdm(glob.glob(os.path.join(args.input, "*.json"))):
            # Skip any subdirectories
            if os.path.isdir(infile):
                continue

            # Optionally skip existing files
            outfile = os.path.join(args.output, os.path.basename(infile))
            if args.skip_existing and os.path.isfile(outfile):
                continue

            try:
                process_file(
                    infile,
                    outfile,
                    args.strip_existing,
                    model,
                    tokenizer,
                    suppress_tokens,
                    device,
                    args,
                )
            except Exception as e:
                sys.stderr.write(f"\nError processing {infile}: {e}\n")


if __name__ == "__main__":
    main()
