import argparse
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
    if obj.get("type") == "object":
        for k, v in obj["properties"].items():
            yield from get_all_paths(v, prefix + ".properties." + k)
    elif obj.get("type") in ["string", "number", "boolean"]:
        yield prefix
    elif obj.get("type") == "array" and "items" in obj:
        yield prefix


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--schema-type",
        type=str,
        default="jsonschema",
        choices=["jsonschema", "typescript", "zod"],
    )
    parser.add_argument("-m", "--model", type=str, default="replit/replit-code-v1-3b")
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

    sys.stderr.write("Generating descriptions…\n")
    descriptions = {}
    for path in tqdm(list(get_all_paths(obj))):
        # Add the description as a tag and use it to find where to remove the
        # tag so we can start description generation after the opening qupte
        desc_path = jsonpath_ng.parse(path).child(jsonpath_ng.Fields("description"))
        desc_obj = desc_path.update_or_create(obj, DESC_TAG)

        if args.schema_type == "jsonschema":
            desc_str = json.dumps(desc_obj)
        elif args.schema_type == "typescript":
            desc_obj["title"] = "JSONSchema"
            out = subprocess.run(
                ["yarn", "json2ts", "--unreachableDefinitions"],
                input=json.dumps(desc_obj),
                capture_output=True,
                encoding="utf-8",
            )
            desc_str = out.stdout
        elif args.schema_type == "zod":
            out = subprocess.run(
                ["yarn", "json-schema-to-zod", "-s", "/dev/stdin"],
                input=json.dumps(desc_obj),
                capture_output=True,
                encoding="utf-8",
            )
            desc_str = out.stdout

        desc_str = desc_str[: desc_str.find(DESC_TAG)]

        # Encode the input and generate a description
        x = tokenizer.encode(desc_str, return_tensors="pt")
        y = model.generate(
            x,
            do_sample=True,
            num_beams=3,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[
                StringStoppingCriteria(tokenizer, len(x), args.schema_type)
            ],
        )
        generated_code = tokenizer.decode(
            y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Clean up the description by stripping whitespace and quote if needed
        desc = generated_code[len(desc_str) :]
        last_quote = desc.rfind('"')
        if last_quote != -1 and desc[last_quote - 1] != "\\":
            desc = desc[:last_quote]
        desc = desc.strip()

        # Store this description to update later
        descriptions[str(desc_path)] = desc

    # Iterate through all the collected descriptions and update the object
    for path, desc in descriptions.items():
        obj = jsonpath_ng.parse(path).update_or_create(obj, desc)

    print(json.dumps(obj, indent=4))


if __name__ == "__main__":
    main()
