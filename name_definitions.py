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

import jsonpath_ng
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

import utils


# Uses code from https://github.com/dpfried/incoder/blob/main/example_usage.py

SPLIT_TOKEN = "<|insert|>"
EOM = "<|endofmask|>"
BOS = "<|endoftext|>"
FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
FIM_INDICATOR = "<FILL_HERE>"


def rename_key(old_name, new_name, reorder=False):
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


def get_defn_paths(obj, prefix="$"):
    # Add descriptions to any top-level definitions
    def_keys = ["definitions", "$defs"]
    for def_key in def_keys:
        if prefix == "$" and def_key in obj:
            for k, v in obj[def_key].items():
                yield prefix + "." + def_key + "." + k


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
    return re.sub("^\$", "#", path).replace(".", "/")


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
    renamed_schema = defn_path.left.update_or_create(
        copy.deepcopy(schema),
        rename_key(str(defn_path.right), token, reorder=True),
    )
    defn = defn_path.left.find(renamed_schema)[0].value

    # Dump the definition as JSON
    return json.dumps({"definitions": defn}, indent=4)[:514]


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


def infill_defn_name(schema, defn_path, model, tokenizer, device):
    defn_str = get_defn_template(schema, defn_path, FIM_INDICATOR)
    # See https://huggingface.co/spaces/bigcode/bigcode-playground/blob/main/app.py
    prefix, suffix = defn_str.split(FIM_INDICATOR)
    defn_str = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    print(defn_str)

    # Encode the input and generate a description
    x = tokenizer.encode(defn_str, return_tensors="pt").to(device)
    y = model.generate(
        x,
        generation_config=GenerationConfig(
            do_sample=True,
            num_beams=3,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[utils.StringStoppingCriteria(tokenizer, len(x))],
        ),
    )
    generated_code = tokenizer.decode(
        y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    y = model.generate(x)
    generated_code = tokenizer.decode(y[0])

    return utils.strip_generated_code(generated_code[len(defn_str) :])


def process_file(infile, outfile, model, tokenizer, device, args):
    with open(infile, "r") as f:
        json_str = f.read()
        obj = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(
            json_str
        )

    paths = list(get_defn_paths(obj))

    # Erase the existing definition names before continuing
    orig_defs = {}
    if not args.keep_existing:
        orig_mapping = {}
        for i, path in enumerate(paths):
            defn_path = jsonpath_ng.parse(path)

            # Rename the definition in the object
            new_name = "defn" + str(i)
            orig_defs[paths[i]] = defn_path.find(obj)[0].value
            obj = defn_path.left.update_or_create(
                obj, rename_key(str(defn_path.right), new_name)
            )

            # Replace the old path with the new path
            old_path = paths[i]
            paths[i] = ".".join(paths[i].split(".")[:-1] + [new_name])
            obj = replace_references(obj, old_path, paths[i])

            # Keep a mapping so we know the original definition name
            orig_mapping[paths[i]] = old_path
    else:
        orig_mapping = {path: path for path in paths}

    defn_names = {}
    new_names = set()
    final_mapping = {}
    for path in tqdm(paths, desc=os.path.basename(infile), leave=False):
        defn_path = jsonpath_ng.parse(path)

        if args.model_name.startswith("bigcode/"):
            defn_name = infill_defn_name(obj, defn_path, model, tokenizer, device)
        else:
            if args.model_name.startswith("hf-internal-testing/"):
                mask_token = "[MASK]"
            else:
                mask_token = "<mask>"
            defn_name = generate_defn_name(
                obj, defn_path, model, tokenizer, mask_token, device
            )

        # Add a numerical suffix if needed
        if defn_name in new_names:
            defn_suffix = 2
            while (defn_name + str(i)) in new_names:
                defn_suffix += 1
            defn_name += str(defn_suffix)
        new_names.add(defn_name)

        # Store this definition name to update later
        defn_names[path] = defn_name
        final_mapping[orig_mapping[path]] = ".".join(path.split(".")[:-1] + [defn_name])

    # Iterate through all the collected descriptions and update the object
    for path, defn_name in defn_names.items():
        defn_path = jsonpath_ng.parse(path)
        obj = defn_path.left.update_or_create(
            copy.deepcopy(obj), rename_key(str(defn_path.right), defn_name)
        )
        new_path = ".".join(path.split(".")[:-1] + [defn_name])
        obj = replace_references(obj, path, new_path)

    # Output the mapping between old and new definitions
    if args.output_mapping:
        writer = csv.writer(sys.stderr)
        for orig, final in final_mapping.items():
            writer.writerow([orig, final])

    with open(outfile, "w") as f:
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
    parser.add_argument("-k", "--keep-existing", default=False, action="store_true")
    parser.add_argument("--output-mapping", default=False, action="store_true")
    parser.add_argument("--better-transformer", default=False, action="store_true")
    parser.add_argument("--skip-existing", default=False, action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Load model
    sys.stderr.write("Loading model…\n")

    if args.model_name.startswith("facebook/incoder-"):
        # Add model-specific parameters
        kwargs = {}

        if args.model_name == "facebook/incoder-6B":
            kwargs["low_cpu_mem_usage"] = True
            if not args.cpu:
                kwargs["revision"] = "float16"
                kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True, device_map="auto", **kwargs
        ).to(device)

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
        ).to(device)

        # Load tokenizer
        sys.stderr.write("Loading tokenizer…\n")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, device_map="auto"
        )
    else:
        model = pipeline("fill-mask", model=args.model_name, device=device)
        tokenizer = None

    # Convert to BetterTransformer
    if args.better_transformer:
        model = model.to_bettertransformer()

    sys.stderr.write("Generating definition names…\n")
    if os.path.isfile(args.input):
        if os.path.isdir(args.output):
            args.output = os.path.join(args.output, os.path.basename(args.input))
        process_file(args.input, args.output, model, tokenizer, device, args)
    elif os.path.isdir(args.input):
        for infile in tqdm(glob.glob(os.path.join(args.input, "*.json"))):
            # Skip any subdirectories
            if os.path.isdir(infile):
                continue

            # Optionally skip existing files
            outfile = os.path.join(args.output, os.path.basename(infile))
            if args.skip_existing and os.path.isfile(outfile):
                continue

            process_file(infile, outfile, model, tokenizer, device, args)


if __name__ == "__main__":
    main()
