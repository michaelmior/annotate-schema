import argparse
import collections
import json
import re
import string

import bert_score
import jsonpath_ng
from moverscore_v2 import word_mover_score
import nltk
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import wordninja

import bart_score
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


def compare_descriptions(obj1, obj2, score_func):
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


def compare_definitions(obj1, obj2, score_func):
    cands = []
    refs = []

    for defn_key in ("definitions", "$defs"):
        cands.extend(obj1.get(defn_key, {}).keys())
        refs.extend(obj2.get(defn_key, {}).keys())
        assert len(cands) == len(refs)

    return score_func(cands, refs)


def score_bleu(cands, refs):
    cands_tok = list(map(split_token, cands))
    refs_tok = [[split_token(s)] for s in refs]
    for a, b in zip(cands_tok, refs_tok):
        print(a, b)
    return nltk.translate.bleu_score.corpus_bleu(refs_tok, cands_tok)


def score_meteor(cands, refs):
    sims = []
    for cand, ref in zip(cands, refs):
        cand_tok = split_token(cand)
        ref_tok = split_token(ref)
        sims.append(nltk.translate.meteor_score.single_meteor_score(ref_tok, cand_tok))

    return sum(sims) / len(sims)


def score_bertscore(cands, refs):
    P, R, F1 = bert_score.score(cands, refs, lang="en", rescale_with_baseline=True)
    return float(F1.mean())


def score_bartscore(cands, refs):
    bs = bart_score.BARTScorer()
    return np.mean(bs.score(refs, cands))


def score_moverscore(cands, refs):
    scores = []
    for r, c in zip(refs, cands):
        idf_dict_hyp = collections.defaultdict(lambda: 1.0)
        idf_dict_ref = collections.defaultdict(lambda: 1.0)
        score = word_mover_score(
            [r],
            [c],
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        )
        scores.append(np.mean(score))
    return np.mean(scores)


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
        choices=["cosine", "bertscore", "bleu", "meteor", "bartscore", "moverscore"],
    )
    parser.add_argument("-d", "--descriptions", default=False, action="store_true")
    parser.add_argument("-n", "--definitions", default=False, action="store_true")
    args = parser.parse_args()

    # Select the desired scorer
    score_func = score_cosine
    if args.scorer == "bertscore":
        score_func = score_bertscore
    elif args.scorer == "bleu":
        score_func = score_bleu
    elif args.scorer == "meteor":
        score_func = score_meteor
    elif args.scorer == "bartscore":
        score_func = score_bartscore
    elif args.scorer == "moverscore":
        score_func = score_moverscore

    # Load both objects
    obj1 = json.load(open(args.candidate))
    obj2 = json.load(open(args.reference))

    # Print similarity
    if args.descriptions:
        print("Descriptions: ", compare_descriptions(obj1, obj2, score_func))
    if args.definitions:
        print(" Definitions: ", compare_definitions(obj1, obj2, score_func))
