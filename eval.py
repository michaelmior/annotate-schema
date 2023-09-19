import argparse
import copy
import json
import re
import string

import jsonpath_ng
import nlgmetricverse
from tqdm import tqdm
import wordninja

from annotate_schema import get_all_paths


# See https://stackoverflow.com/a/29920015
def camel_case_split(identifier):
    matches = re.finditer(
        ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier
    )
    return [m.group(0) for m in matches]


def split_token(token):
    # Try splitting by underscores
    if "_" in token:
        return token.split("_")
    # Try splitting by hyphens
    elif "-" in token:
        return token.split("_")
    # If camelCase, try splitting by capitals
    elif any(c in string.ascii_lowercase for c in token) and any(
        c in string.ascii_uppercase for c in token
    ):
        # Split camelCase tokens
        return camel_case_split(token)
    # Otherwise, we have all lower (or upper) case letters,
    # so try to split probabilistically into English words
    else:
        return wordninja.split(token)


def compare_descriptions(obj1, obj2, scorer):
    cands = []
    refs = []

    paths = get_all_paths(obj1)
    for path in tqdm(list(paths)):
        desc_path = jsonpath_ng.parse(path).child(jsonpath_ng.Fields("description"))

        # Check if the original schema has a description
        orig_desc = desc_path.find(obj2)
        if len(orig_desc) == 0:
            continue

        cands.append(desc_path.find(obj1)[0].value)
        refs.append(desc_path.find(obj2)[0].value)

    return scorer(predictions=cands, references=refs)


def print_scores(header, metrics, scores):
    print(header)
    for metric in metrics:
        print(scores[metric])
        if metric == "nubia":
            score_key = "nubia_score"
        else:
            score_key = "score"
        print("  ", metric, ":", scores[metric][score_key])


def compare_definitions(obj1, obj2, scorer):
    cands = []
    refs = []

    for defn_key in ("definitions", "$defs"):
        cands.extend(obj1.get(defn_key, {}).keys())
        refs.extend(obj2.get(defn_key, {}).keys())
        assert len(cands) == len(refs)

    return scorer(predictions=cands, references=refs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument(
        "-s",
        "--scorers",
        type=lambda t: [s.strip() for s in t.split(",")],
        required=True,
    )
    parser.add_argument("-d", "--descriptions", default=False, action="store_true")
    parser.add_argument("-n", "--definitions", default=False, action="store_true")
    args = parser.parse_args()

    # Select the desired metrics
    scorers = copy.deepcopy(args.scorers)
    scorer = nlgmetricverse.NLGMetricverse(metrics=scorers)

    # Load both objects
    obj1 = json.load(open(args.candidate))
    obj2 = json.load(open(args.reference))

    # Print similarity
    if args.descriptions:
        scores = compare_descriptions(obj1, obj2, scorer)
        print_scores("Descriptions", args.scorers, scores)
    if args.definitions:
        scores = compare_definitions(obj1, obj2, scorer)
        print_scores("Definitions", args.scorers, scores)
