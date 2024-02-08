import argparse
import os

import jsonpath_ng
import torch
from transformers import PreTrainedTokenizer, StoppingCriteria


# Adapted from jsonformer
# https://github.com/1rgs/jsonformer/blob/bfad031876ace84ec0a7853718a1c0828ea1653a/jsonformer/logits_processors.py#L5-L23
class StringStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        schema_type: str = "jsonschema",
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

        last_token_id = input_ids[0][-1:]
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


def strip_generated_code(code):
    # Clean up the description by stripping whitespace and quote if needed
    last_quote = code.rfind('"')
    if last_quote != -1 and code[last_quote - 1] != "\\":
        code = code[:last_quote]

    # Clean up and return the generated description
    return code.split('"}')[0].strip()


# See https://stackoverflow.com/a/33181083/123695
class InputOutputType:
    def __init__(self, is_input=True):
        self._is_input = is_input

    def __call__(self, path):
        if path == "-":
            if self._is_input:
                return "/dev/stdin"
            else:
                return "/dev/stdout"
        else:
            if self._is_input:
                if not os.path.exists(path):
                    raise argparse.ArgumentTypeError(
                        "input path does not exist: '%s'" % path
                    )
            else:
                parent = os.path.dirname(os.path.normpath(path)) or "."
                if not os.path.isdir(parent):
                    raise argparse.ArgumentTypeError(
                        "parent path is not a directory: '%s'" % parent
                    )
                elif not os.path.exists(parent):
                    raise argparse.ArgumentTypeError(
                        "parent directory does not exist: '%s'" % parent
                    )

        return path


def strip_meta(obj, paths):
    for path in paths:
        strip_path = path.child(jsonpath_ng.Fields("title", "description", "$comment"))
        obj = strip_path.filter(lambda _: True, obj)

    return obj


def get_all_paths(obj, prefix=jsonpath_ng.Root()):
    # Skip anything not a dictionary
    if not isinstance(obj, dict):
        return

    # Add descriptions to any top-level definitions
    def_keys = ["definitions", "$defs"]
    for def_key in def_keys:
        if isinstance(prefix, jsonpath_ng.Root) and def_key in obj:
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

    prop_keys = ["properties", "patternProperties", "additionalProperties"]
    for prop_key in prop_keys:
        prop_obj = obj.get(prop_key, {})
        # additionalProperties can be a Boolean, so we check
        if not isinstance(prop_obj, dict):
            continue

        for k, v in prop_obj.items():
            yield from get_all_paths(
                v,
                jsonpath_ng.Child(
                    prefix,
                    jsonpath_ng.Child(
                        jsonpath_ng.Fields(prop_key), jsonpath_ng.Fields(k)
                    ),
                ),
            )

    if (
        obj.get("type") in ["object", "integer", "string", "number", "boolean"]
        or "$ref" in obj
    ):
        yield prefix
    elif obj.get("type") == "array" and "items" in obj:
        yield prefix
        yield from get_all_paths(
            obj["items"], jsonpath_ng.Child(prefix, jsonpath_ng.Fields("items"))
        )

    for k in ("allOf", "anyOf", "oneOf"):
        yield prefix
        if k in obj:
            for i, v in enumerate(obj[k]):
                yield from get_all_paths(
                    v,
                    jsonpath_ng.Child(
                        prefix,
                        jsonpath_ng.Child(jsonpath_ng.Fields(k), jsonpath_ng.Index(i)),
                    ),
                )
