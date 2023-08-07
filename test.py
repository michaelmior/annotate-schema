import argparse
import json

import bert_score
import jsonpath_ng
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from annotate_schema import get_all_paths


def compare_objects(obj1, obj2, score_func):
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

    return score_func(cands, refs)


def score_bertscore(cands, refs):
    P, R, F1 = bert_score.score(cands, refs, lang="en", rescale_with_baseline=True)
    return float(F1.mean())


def score_cosine(cands, refs):
    # Load the pretrained tokenizer and embedding models
    tokenizer = AutoTokenizer.from_pretrained(
        "princeton-nlp/sup-simcse-bert-base-uncased"
    )
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    sims = []
    for cand, ref in zip(cands, refs):
        descs = [cand, ref]
        inputs = tokenizer(descs, padding=True, truncation=True, return_tensors="pt")

        # Find the embeddings of both descriptions
        with torch.no_grad():
            embeddings = model(
                **inputs, output_hidden_states=True, return_dict=True
            ).pooler_output

        # Add to the list of collected similarities
        cosine_sim = 1 - cosine(embeddings[0], embeddings[1])
        sims.append(cosine_sim)

    return sum(sims) / len(sims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("reference")
    parser.add_argument(
        "-s",
        "--scorer",
        type=str,
        default="cosine",
        choices=["cosine", "bertscore"],
    )
    args = parser.parse_args()

    # Select the desired scorer
    score_func = score_cosine
    if args.scorer == "bertscore":
        score_func = score_bertscore

    # Load both objects
    obj1 = json.load(open(args.candidate))
    obj2 = json.load(open(args.reference))

    # Print similarity
    print(compare_objects(obj1, obj2, score_func))
