import argparse
import copy
import glob
import json
import os
import subprocess
import sys

import json5
import jsonpath_ng
import peft
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from auto_gptq import AutoGPTQForCausalLM

from utils import (
    get_all_paths,
    strip_meta,
    strip_generated_code,
    InputOutputType,
    StringStoppingCriteria,
)

DESC_TAG = "!!!DESCRIPTION!!!"
YARN_CMD = ["yarn", "run", "--silent", "--"]


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
    schema,
    desc_path,
    schema_type,
    model,
    tokenizer,
    device="cpu",
    max_tokens=2048,
    num_beams=3,
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
            num_beams=num_beams,
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
    with open(infile, "r", encoding="utf-8") as f:
        json_str = f.read()
        obj = json5.loads(json_str)

    assert DESC_TAG not in json_str

    paths = list(get_all_paths(obj))

    # Strip existing descriptions if requested
    if args.strip_existing:
        obj = strip_meta(obj, paths)

    descriptions = {}
    for path in tqdm(paths, desc=os.path.basename(infile), leave=False):
        desc_path = path.child(jsonpath_ng.Fields("description"))
        desc = generate_description(
            obj,
            desc_path,
            args.schema_type,
            model,
            tokenizer,
            device,
            args.max_tokens,
            args.num_beams,
        )

        # Store this description to update later
        descriptions[desc_path] = desc

    # Iterate through all the collected descriptions and update the object
    for path, desc in descriptions.items():
        obj = path.update_or_create(obj, desc)

    with open(outfile, "w", encoding="utf-8") as f:
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
    parser.add_argument("-m", "--model", type=str, default="replit/replit-code-v1_5-3b")
    parser.add_argument(
        "-b", "--model-basename", dest="basename", type=str, default=None
    )
    parser.add_argument("-t", "--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--no-strip-existing", dest="strip_existing", default=True, action="store_false"
    )
    parser.add_argument("-c", "--cpu", default=False, action="store_true")
    parser.add_argument("-d", "--device-map-auto", default=False, action="store_true")
    parser.add_argument("--better-transformer", default=False, action="store_true")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--skip-existing", default=False, action="store_true")
    parser.add_argument("-4", "--load-in-4bit", default=False, action="store_true")
    parser.add_argument("-8", "--load-in-8bit", default=False, action="store_true")
    parser.add_argument("-p", "--paged", default=False, action="store_true")
    parser.add_argument("-q", "--qlora", default=False, action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    # Enable settings to be used with QLora
    if args.qlora:
        args.load_in_4bit = True
        args.paged = True

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("Only one of --load-in-4bit or --load-in-8bit can be used")

    if os.path.isdir(args.input):
        if os.path.exists(args.output):
            if not os.path.isdir(args.output):
                parser.error(
                    f"Output {args.output} already exists and is not a directory"
                )
        else:
            os.makedirs(args.output)

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Load model
    sys.stderr.write("Loading model…\n")

    # Construct the quantization configuration
    qkwargs = {}
    if args.load_in_4bit:
        qkwargs["load_in_4bit"] = True
    if args.load_in_8bit:
        qkwargs["load_in_8bit"] = True
    if args.qlora:
        qkwargs["bnb_4bit_use_double_quant"] = True
        qkwargs["bnb_4bit_quant_type"] = "nf4"
        qkwargs["bnb_4bit_compute_dtype"] = torch.float16
    qconfig = BitsAndBytesConfig(**qkwargs)

    # Add model-specific parameters
    kwargs = {}
    if args.model.startswith("facebook/incoder-"):
        kwargs["low_cpu_mem_usage"] = True
        if not args.cpu and torch.cuda.is_available():
            kwargs["revision"] = "float16"
            kwargs["torch_dtype"] = torch.float16

    if args.load_in_4bit or args.load_in_8bit:
        kwargs["quantization_config"] = qconfig

    if args.device_map_auto:
        kwargs["device_map"] = "auto"

    if args.model.endswith("GPTQ"):
        model = AutoGPTQForCausalLM.from_quantized(
            args.model,
            model_basename=args.basename,
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
            quantize_config=None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, **kwargs
        )

    # If not using a quantized model, set to the correct device
    if not args.load_in_4bit and not args.load_in_8bit:
        model = model.to(device)

    # Convert to BetterTransformer
    if args.better_transformer:
        model = model.to_bettertransformer()

    if args.checkpoint:
        model = peft.PeftModel.from_pretrained(model, model_id=args.checkpoint)

    # load tokenizer
    sys.stderr.write("Loading tokenizer…\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map="auto"
    )

    sys.stderr.write("Generating descriptions…\n")
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
