import argparse
import json

import bert_score
import jsonpath_ng
import nltk
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from annotate_schema import get_all_paths


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
    cands_tok = list(map(nltk.tokenize.word_tokenize, cands))
    refs_tok = [[nltk.tokenize.word_tokenize(s)] for s in refs]
    return nltk.translate.bleu_score.corpus_bleu(refs_tok, cands_tok)


def score_meteor(cands, refs):
    sims = []
    for cand, ref in zip(cands, refs):
        cand_tok = nltk.tokenize.word_tokenize(cand)
        ref_tok = nltk.tokenize.word_tokenize(ref)
        sims.append(nltk.translate.meteor_score.single_meteor_score(ref_tok, cand_tok))

    return sum(sims) / len(sims)


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
        choices=["cosine", "bertscore", "bleu", "meteor"],
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

    # Load both objects
    obj1 = json.load(open(args.candidate))
    obj2 = json.load(open(args.reference))

    # Print similarity
    if args.descriptions:
        print("Descriptions: ", compare_descriptions(obj1, obj2, score_func))
    if args.definitions:
        print(" Definitions: ", compare_definitions(obj1, obj2, score_func))
