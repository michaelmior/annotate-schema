import collections
import copy
import glob
import json
import random
import sys

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


def find_type_paths(obj, json_type, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            # If we have the type keyword and it's value matches, we found one
            if k == "type" and v == json_type:
                yield path

            # Continue recursively through the object's children
            yield from find_type_paths(v, json_type, path + '."' + k + '"')
    elif isinstance(obj, list):
        # Check each list element
        for i, v in enumerate(obj):
            yield from find_type_paths(v, json_type, path + "[" + str(i) + "]")


def write_obj(obj, keyword, is_neg):
    # Remove the description since we generally
    # won't have this at inference time
    if "description" in obj:
        obj = copy.deepcopy(obj)
        obj.pop("description")

    # Skip any objects that are too large
    if len(json.dumps(obj, indent=4)) <= 1024:
        print(json.dumps({"obj": obj, "keyword": keyword, "is_neg": is_neg}))


# Loop over all schemas
if len(sys.argv) < 2:
    schemas = glob.glob("schemas/*.json")
else:
    schemas = sys.argv[1:]

for file in tqdm.tqdm(schemas):
    with open(file) as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError:
            sys.stderr.write("Invalid JSON: " + file + "\n")
            continue

        neg_pos = collections.defaultdict(lambda: ([], []))
        for keyword_type, keywords in KEYWORDS.items():
            # Find all schema elements with the given type
            found_paths = find_type_paths(data, keyword_type)
            for found_path in found_paths:
                try:
                    path = jsonpath_ng.parse(found_path).find(data)
                except jsonpath_ng.exceptions.JsonPathLexerError:
                    # This path is invalid, probably because it's not properly
                    # escaped. We ignore for now since this is rare.
                    sys.stderr.write("Invalid path: " + found_path + "\n")
                    continue

                for found_obj in path:
                    # Get the name of the key which is the last part of the path
                    name = found_path.split(".")[-1]
                    found_obj = {name: found_obj.value}

                    # For each keyword, check if it is included
                    for keyword in keywords:
                        # Append to the 1st or 2nd list depending on
                        # whether the keyword is used in this object
                        has_keyword = keyword in next(iter(found_obj.values()))
                        neg_pos[keyword][has_keyword].append(found_obj)

        for keyword, (neg, pos) in neg_pos.items():
            # Only include cases where we have at least
            # two positive and negative examples
            if len(pos) > 1 and len(neg) > 1:
                # Write out the positive and negative examples
                for pos_item in pos:
                    # Here we need to remove the keyword being trained on
                    item_copy = copy.deepcopy(pos_item)
                    next(iter(item_copy.values())).pop(keyword)

                    write_obj(item_copy, keyword, False)

                # For negative items, generate at most the number of positives
                for neg_item in random.sample(neg, min(len(pos), len(neg))):
                    write_obj(neg_item, keyword, True)
