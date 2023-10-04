import argparse
import copy
import json
import os
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

from utils import strip_generated_code, InputOutputType, StringStoppingCriteria

DESC_TAG = "!!!DESCRIPTION!!!"
YARN_CMD = ["yarn", "run", "--silent", "--"]


def get_all_paths(obj, prefix=jsonpath_ng.Root()):
    # Skip anything not a dictionary
    if not isinstance(obj, dict):
        return

    # Add descriptions to any top-level definitions
    def_keys = ["definitions", "$defs"]
    for def_key in def_keys:
        if prefix == "$" and def_key in obj:
            for k, v in obj[def_key].items():
                yield from get_all_paths(
                    v,
                    jsonpath_ng.Child(
                        prefix,
                        jsonpath_ng.Child(
                            jsonpath_ng.Fields(def_key), jsonpath_ng.Fields(k)
                        ),
                    ),
                )

    # TODO: We should be able to have patternProperties here, but
    #       jsonpath_ng doesn't like keys with characters such as ^ or $
    prop_keys = ["properties"]
    if obj.get("type") == "object":
        found_props = False
        for prop_key in prop_keys:
            for k, v in obj.get(prop_key, {}).items():
                found_props = True
                yield from get_all_paths(
                    v,
                    jsonpath_ng.Child(
                        prefix,
                        jsonpath_ng.Child(
                            jsonpath_ng.Fields(prop_key), jsonpath_ng.Fields(k)
                        ),
                    ),
                )

        if not found_props:
            yield prefix

    elif obj.get("type") in ["string", "number", "boolean"] or "$ref" in obj:
        yield prefix
    elif obj.get("type") == "array" and "items" in obj:
        yield prefix
        yield from get_all_paths(
            v, jsonpath_ng.Child(prefix, jsonpath_ng.Fields("items"))
        )

    for k in ("allOf", "anyOf", "oneOf"):
        if k in obj:
            for i, v in enumerate(obj[k]):
                yield from get_all_paths(
                    v,
                    jsonpath_ng.Child(
                        prefix,
                        jsonpath_ng.Child(jsonpath_ng.Fields(k), jsonpath_ng.Index(i)),
                    ),
                )


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


def process_file(infile, outfile, model, tokenizer, device, args):
    with open(infile, "r") as f:
        json_str = f.read()
        obj = json.loads(json_str)

    assert DESC_TAG not in json_str

    paths = list(get_all_paths(obj))

    # Strip existing descriptions if requested
    if args.strip_existing:
        for path in paths:
            desc_path = path.child(jsonpath_ng.Fields("description"))
            obj = desc_path.filter(lambda _: True, obj)

    sys.stderr.write("Generating descriptions…\n")
    descriptions = {}
    for path in tqdm(paths):
        desc_path = path.child(jsonpath_ng.Fields("description"))
        desc = generate_description(
            obj,
            desc_path,
            args.schema_type,
            model,
            tokenizer,
            device,
            args.max_tokens,
        )

        # Store this description to update later
        descriptions[str(desc_path)] = desc

    # Iterate through all the collected descriptions and update the object
    for path, desc in descriptions.items():
        obj = jsonpath_ng.parse(path).update_or_create(obj, desc)

    with open(outfile, "w") as f:
        json.dump(obj, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=InputOutputType(True), default="/dev/stdin"
    )
    parser.add_argument(
        "-o", "--output", type=InputOutputType(False), default="/dev/stdout"
    )
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
    parser.add_argument("-d", "--device-map-auto", default=False, action="store_true")
    parser.add_argument("-8", "--load-in-8bit", default=False, action="store_true")
    parser.add_argument("--better-transformer", default=False, action="store_true")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Load model
    sys.stderr.write("Loading model…\n")

    # Add model-specific parameters
    kwargs = {}
    if args.model.startswith("facebook/incoder-"):
        kwargs["low_cpu_mem_usage"] = True
        if not args.cpu and torch.cuda.is_available():
            kwargs["revision"] = "float16"
            kwargs["torch_dtype"] = torch.float16

    if args.device_map_auto:
        kwargs["device_map"] = "auto"
    if args.load_in_8bit:
        kwargs["load_in_8bit"] = True

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
            args.model, trust_remote_code=True, **kwargs
        ).to(device)

    # Convert to BetterTransformer
    if args.better_transformer:
        model = model.to_bettertransformer()

    # load tokenizer
    sys.stderr.write("Loading tokenizer…\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map="auto"
    )

    if os.path.isfile(args.input):
        if os.path.isdir(args.output):
            args.output = os.path.join(args.output, os.path.basename(args.input))
        process_file(args.input, args.output, model, tokenizer, device, args)
    elif os.path.isdir(args.input):
        for f in os.listdir(args.input):
            infile = os.path.join(args.input, f)
            outfile = os.path.join(args.output, f)
            process_file(infile, outfile, model, tokenizer, device, args)


if __name__ == "__main__":
    main()
