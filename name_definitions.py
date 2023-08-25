import argparse
import collections
import copy
import json
import random
import sys
from typing import List

import jsonpath_ng
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Uses code from https://github.com/dpfried/incoder/blob/main/example_usage.py

SPLIT_TOKEN = "<|insert|>"
EOM = "<|endofmask|>"
BOS = "<|endoftext|>"


def rename_key(old_name, new_name):
    def rename_fn(_data_field, data, field):
        # Shuffle the items since the order shouldn't matter anyway.
        # This also helps make sure a diverse set of possible other
        # definitions are seen when we have to truncate for length.
        random_items = random.sample(list(data[field].items()), len(data[field]))

        # Build an OrderedDict to ensure the definition we're naming is first
        # This avoids it being truncated when the block of definitions is large
        data[field] = collections.OrderedDict(random_items)
        data[field][new_name] = data[field].pop(old_name)
        data[field].move_to_end(new_name, last=False)

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


def generate_defn_name(schema, defn_path, model, tokenizer, device):
    # Rename the definition with our sentinel token
    renamed_schema = defn_path.left.update_or_create(
        copy.deepcopy(schema), rename_key(str(defn_path.right), SPLIT_TOKEN)
    )
    defn = defn_path.left.find(renamed_schema)[0].value

    # We truncate the serialized JSON below to fit the model size
    defn_str = json.dumps({"definitions": defn}, indent=4)[:2048]

    out = infill(model, tokenizer, device, defn_str.split(SPLIT_TOKEN))
    new_defn_name = out["infills"][0]

    # If a quote appears, keep anything before the quote
    if '"' in new_defn_name:
        quote_index = new_defn_name.index('"')
        new_defn_name = new_defn_name[:quote_index]

    return new_defn_name.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", default=False, action="store_true")
    parser.add_argument("-c", "--cpu", default=False, action="store_true")
    args = parser.parse_args()

    if args.small:
        model_name = "facebook/incoder-1B"
    else:
        model_name = "facebook/incoder-6B"

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    json_str = sys.stdin.read()
    obj = json.loads(json_str)

    # Load model
    sys.stderr.write("Loading model…\n")

    # Add model-specific parameters
    kwargs = {}
    if not args.small:
        kwargs["low_cpu_mem_usage"] = True
        if not args.cpu:
            kwargs["revision"] = "float16"
            kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto", **kwargs
    ).to(device)

    # load tokenizer
    sys.stderr.write("Loading tokenizer…\n")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto"
    )
    tokenizer.pad_token = "<pad>"
    tokenizer.padding_side = "left"

    paths = list(get_defn_paths(obj))

    sys.stderr.write("Generating definition names…\n")
    defn_names = {}
    for path in tqdm(paths):
        defn_path = jsonpath_ng.parse(path)
        defn_name = generate_defn_name(obj, defn_path, model, tokenizer, device)

        # Store this definition name to update later
        defn_names[str(defn_path)] = defn_name

    # Iterate through all the collected descriptions and update the object
    for path, defn_name in defn_names.items():
        defn_path = jsonpath_ng.parse(path)
        obj = defn_path.left.update_or_create(
            copy.deepcopy(obj), rename_key(str(defn_path.right), defn_name)
        )

    print(json.dumps(obj, indent=4))


if __name__ == "__main__":
    main()
