import argparse
import collections
import copy
import glob
import json
import random
import sys

import json5
import jsonpath_ng.ext
import tqdm

KEYWORDS = {
    "integer": [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ],
    "object": [
        "minProperties",
        "maxProperties",
    ],
    "string": [
        "minLength",
        "maxLength",
        "format",
        "pattern",
    ],
    "array": [
        "minContains",
        "maxContains",
        "minItems",
        "maxItems",
        "uniqueItems",
    ],
}

# Numeric and integer types use the same keywords
KEYWORDS["numeric"] = KEYWORDS["integer"]


def find_type_paths(obj, json_type, path=jsonpath_ng.Root()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            # If we have the type keyword and it's value matches, we found one
            if k == "type" and v == json_type:
                yield path

            # Continue recursively through the object's children
            yield from find_type_paths(
                v, json_type, jsonpath_ng.Child(path, jsonpath_ng.Fields(k))
            )
    elif isinstance(obj, list):
        # Check each list element
        for i, v in enumerate(obj):
            yield from find_type_paths(
                v, json_type, jsonpath_ng.Child(path, jsonpath_ng.Index(i))
            )


def write_obj(obj, keyword, is_neg):
    # Skip any objects that are too large
    if len(json.dumps(obj, indent=4)) <= 1024:
        print(json.dumps({"obj": obj, "keyword": keyword, "is_neg": is_neg}))


def build_obj(path, value):
    # Build an object with the same shape it has in the entire document
    # For example, a path of $.foo.bar[0] would create an object
    # that looks like {"foo": {"bar": [value]}}
    if isinstance(path, jsonpath_ng.Root):
        return value
    if isinstance(path.right, (jsonpath_ng.Index, jsonpath_ng.Slice)):
        return build_obj(path.left, [value])
    elif isinstance(path.right, jsonpath_ng.Fields):
        return build_obj(path.left, {path.right.fields[0]: value})
    else:
        return value


def get_examples(data):
    neg_pos = collections.defaultdict(lambda: ([], []))
    for keyword_type, keywords in KEYWORDS.items():
        # Find all schema elements with the given type
        found_paths = find_type_paths(data, keyword_type)
        for found_path in found_paths:
            for found_obj in found_path.find(data):
                # For each keyword, check if it is included
                for keyword in keywords:
                    # Append to the 1st or 2nd list depending on
                    # whether the keyword is used in this object
                    has_keyword = keyword in found_obj.value
                    obj_without_keyword = copy.deepcopy(found_obj.value)

                    # Remove the keyword since we can't use it to predict
                    obj_without_keyword.pop(keyword, None)

                    # Remove the description since we generally
                    # won't have this at inference time
                    obj_without_keyword.pop("description", None)

                    path_obj = build_obj(found_path, obj_without_keyword)
                    neg_pos[keyword][has_keyword].append(path_obj)

    return neg_pos


def write_examples(neg_pos):
    for keyword, (neg, pos) in neg_pos.items():
        # Only include cases where we have at least
        # two positive and negative examples
        if len(pos) > 1 and len(neg) > 1:
            # Write out the positive and negative examples
            for pos_item in pos:
                write_obj(pos_item, keyword, False)

            # For negative items, generate at most the number of positives
            for neg_item in random.sample(neg, min(len(pos), len(neg))):
                write_obj(neg_item, keyword, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("schemas", nargs="*", metavar="schema")
    args = parser.parse_args()

    # Loop over all schemas
    if args.schemas:
        schemas = args.schemas
    else:
        schemas = glob.glob("schemas/*.json")

    for file in tqdm.tqdm(schemas):
        with open(file) as f:
            try:
                data = json5.load(f)
            except json.decoder.JSONDecodeError:
                sys.stderr.write("Invalid JSON: " + file + "\n")
                continue

            neg_pos = get_examples(data)
            write_examples(neg_pos)


if __name__ == "__main__":
    main()
