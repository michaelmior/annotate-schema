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
    GenerationConfig,
)
from auto_gptq import AutoGPTQForCausalLM

from utils import strip_generated_code, StringStoppingCriteria

DESC_TAG = "!!!DESCRIPTION!!!"
YARN_CMD = ["yarn", "run", "--silent", "--"]


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

    elif obj.get("type") in ["string", "number", "boolean"] or "$ref" in obj:
        yield prefix
    elif obj.get("type") == "array" and "items" in obj:
        yield prefix
        yield from get_all_paths(obj["items"], prefix + ".items")

    for k in ("allOf", "anyOf", "oneOf"):
        if k in obj:
            for i, v in enumerate(obj[k]):
                yield from get_all_paths(v, prefix + "." + k + "[" + str(i) + "]")


def convert_schema(obj, schema_type):
    if schema_type == "jsonschema":
        desc_str = json.dumps(obj)
    elif schema_type == "pydantic":
        out = subprocess.run(
            ["pipenv", "run", "datamodel-codegen", "--use-double-quotes"],
            input=json.dumps(obj),
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        desc_str = out.stdout
    elif schema_type == "typescript":
        obj["title"] = "JSONSchema"
        out = subprocess.run(
            YARN_CMD + ["json2ts", "--unreachableDefinitions"],
            input=json.dumps(obj),
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        desc_str = out.stdout
    elif schema_type == "zod":
        out = subprocess.run(
            YARN_CMD + ["json-schema-to-zod", "-s", "/dev/stdin"],
            input=json.dumps(obj),
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        desc_str = out.stdout

    return desc_str


def generate_description(
    schema, desc_path, schema_type, model, tokenizer, device="cpu", max_tokens=2048
):
    # Add the description as a tag and use it to find where to remove the
    # tag so we can start description generation after the opening quote
    desc_obj = desc_path.update_or_create(copy.deepcopy(schema), DESC_TAG)

    # Convert the schema to the approprate format
    desc_str = convert_schema(desc_obj, schema_type)

    # Strip out everything after the fake description that was inserted
    desc_str = desc_str[: desc_str.find(DESC_TAG)]

    # Only use up to the given maximum number of tokens
    desc_str = desc_str[-max_tokens:]

    # Encode the input and generate a description
    x = tokenizer.encode(desc_str, return_tensors="pt").to(device)
    y = model.generate(
        x,
        generation_config=GenerationConfig(
            do_sample=True,
            num_beams=3,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
        ),
        stopping_criteria=[StringStoppingCriteria(tokenizer, len(x), schema_type)],
    )
    generated_code = tokenizer.decode(
        y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return strip_generated_code(generated_code[len(desc_str) :])


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
    parser.add_argument(
        "-b", "--model-basename", dest="basename", type=str, default=None
    )
    parser.add_argument("-t", "--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--no-strip-existing", dest="strip_existing", default=True, action="store_false"
    )
    parser.add_argument("-c", "--cpu", default=False, action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    json_str = sys.stdin.read()
    obj = json.loads(json_str)

    assert DESC_TAG not in json_str

    # Load model
    sys.stderr.write("Loading model…\n")

    # Add model-specific parameters
    kwargs = {}
    if args.model.startswith("facebook/incoder-"):
        kwargs["low_cpu_mem_usage"] = True
        if not args.cpu and torch.cuda.is_available():
            kwargs["revision"] = "float16"
            kwargs["torch_dtype"] = torch.float16

    if args.model.endswith("GPTQ"):
        model = AutoGPTQForCausalLM.from_quantized(
            args.model,
            model_basename=args.basename,
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
            quantize_config=None,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, device_map="auto", **kwargs
        ).to(device)

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
            obj, desc_path, args.schema_type, model, tokenizer, device, args.max_tokens
        )

        # Store this description to update later
        descriptions[str(desc_path)] = desc

    # Iterate through all the collected descriptions and update the object
    for path, desc in descriptions.items():
        obj = jsonpath_ng.parse(path).update_or_create(obj, desc)

    print(json.dumps(obj, indent=4))


if __name__ == "__main__":
    main()
