import argparse
import copy
import json
import subprocess
import sys

import jsonpath_ng
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    StoppingCriteria,
)

DESC_TAG = "!!!DESCRIPTION!!!"


def get_all_paths(obj, prefix="$"):
    # Add descriptions to any top-level definitions
    def_keys = ["definitions", "$defs"]
    for def_key in def_keys:
        if prefix == "$" and def_key in obj:
            for k, v in obj[def_key].items():
                yield from get_all_paths(v, prefix + "." + def_key + "." + k)

    # TODO: We should be able to have patternProperties here, but
    #       jsonpath_ng doesn't like keys with characters such as ^ or $
    prop_keys = ["properties"]
    if obj.get("type") == "object":
        found_props = False
        for prop_key in prop_keys:
            for k, v in obj.get(prop_key, {}).items():
                found_props = True
                yield from get_all_paths(v, prefix + "." + prop_key + "." + k)

        if not found_props:
            yield prefix

    elif obj.get("type") in ["string", "number", "boolean"] or '$ref' in obj:
        yield prefix
    elif obj.get("type") == "array" and "items" in obj:
        yield prefix
        yield from get_all_paths(obj["items"], prefix + ".items")

    for k in ("allOf", "anyOf", "oneOf"):
        if k in obj:
            for i, v in enumerate(obj[k]):
                yield from get_all_paths(v, prefix + "." + k + "[" + str(i) + "]")


# Adapted from jsonformer
# https://github.com/1rgs/jsonformer/blob/bfad031876ace84ec0a7853718a1c0828ea1653a/jsonformer/logits_processors.py#L5-L23
class StringStoppingCriteria(StoppingCriteria):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, prompt_length: int, schema_type: str
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.schema_type = schema_type

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _,
    ) -> bool:
        if len(input_ids[0]) <= self.prompt_length:
            return False

        last_token_id = input_ids[0][-1]
        last_token = self.tokenizer.decode(last_token_id, skip_special_tokens=True)

        if self.schema_type == "typescript":
            return "*/" in last_token
        else:
            # Check if there is a quote in the last token
            last_quote = last_token.rfind('"')
            if last_quote == -1:
                return False
            else:
                # If there is a quote, make sure it isn't escaped
                return last_token[last_quote - 1] != "\\"


def generate_description(schema, desc_path, schema_type, model, tokenizer, max_tokens):
    # Add the description as a tag and use it to find where to remove the
    # tag so we can start description generation after the opening qupte
    desc_obj = desc_path.update_or_create(copy.deepcopy(schema), DESC_TAG)

    if schema_type == "jsonschema":
        desc_str = json.dumps(desc_obj)
    elif schema_type == "pydantic":
        out = subprocess.run(
            ["pipenv", "run", "datamodel-codegen", "--use-double-quotes"],
            input=json.dumps(desc_obj),
            capture_output=True,
            encoding="utf-8",
        )
        desc_str = out.stdout
    elif schema_type == "typescript":
        desc_obj["title"] = "JSONSchema"
        out = subprocess.run(
            ["yarn", "json2ts", "--unreachableDefinitions"],
            input=json.dumps(desc_obj),
            capture_output=True,
            encoding="utf-8",
        )
        desc_str = out.stdout
    elif schema_type == "zod":
        out = subprocess.run(
            ["yarn", "json-schema-to-zod", "-s", "/dev/stdin"],
            input=json.dumps(desc_obj),
            capture_output=True,
            encoding="utf-8",
        )
        desc_str = out.stdout

    desc_str = desc_str[: desc_str.find(DESC_TAG)]
    desc_str = desc_str[-max_tokens:]

    # Encode the input and generate a description
    x = tokenizer.encode(desc_str, return_tensors="pt")
    y = model.generate(
        x,
        do_sample=True,
        num_beams=3,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=[StringStoppingCriteria(tokenizer, len(x), schema_type)],
    )
    generated_code = tokenizer.decode(
        y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Clean up the description by stripping whitespace and quote if needed
    desc = generated_code[len(desc_str) :]
    last_quote = desc.rfind('"')
    if last_quote != -1 and desc[last_quote - 1] != "\\":
        desc = desc[:last_quote]

    return desc.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--schema-type",
        type=str,
        default="jsonschema",
        choices=["jsonschema", "pydantic", "typescript", "zod"],
    )
    parser.add_argument("-m", "--model", type=str, default="replit/replit-code-v1-3b")
    parser.add_argument("-t", "--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--no-strip-existing", dest="strip_existing", default=True, action="store_false"
    )
    args = parser.parse_args()

    json_str = sys.stdin.read()
    obj = json.loads(json_str)

    assert DESC_TAG not in json_str

    # Load model
    sys.stderr.write("Loading model…\n")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map="auto"
    )

    # load tokenizer
    sys.stderr.write("Loading tokenizer…\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map="auto"
    )

    paths = list(get_all_paths(obj))

    # Strip existing descriptions if requested
    if args.strip_existing:
        for path in paths:
            desc_path = jsonpath_ng.parse(path).child(jsonpath_ng.Fields("description"))
            obj = desc_path.filter(lambda _: True, obj)

    sys.stderr.write("Generating descriptions…\n")
    descriptions = {}
    for path in tqdm(paths):
        desc_path = jsonpath_ng.parse(path).child(jsonpath_ng.Fields("description"))
        desc = generate_description(
            obj, desc_path, args.schema_type, model, tokenizer, args.max_tokens
        )

        # Store this description to update later
        descriptions[str(desc_path)] = desc

    # Iterate through all the collected descriptions and update the object
    for path, desc in descriptions.items():
        obj = jsonpath_ng.parse(path).update_or_create(obj, desc)

    print(json.dumps(obj, indent=4))


if __name__ == "__main__":
    main()
