import json
import sys

import jsonpath_ng
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from annotate_schema import get_all_paths


def compare_objects(obj1, obj2):
    paths = get_all_paths(obj1)

    # Load the pretrained tokenizer and embedding models
    tokenizer = AutoTokenizer.from_pretrained(
        "princeton-nlp/sup-simcse-bert-base-uncased"
    )
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

    sims = []
    for path in tqdm(list(paths)):
        desc_path = jsonpath_ng.parse(path).child(jsonpath_ng.Fields("description"))

        # Check if the original schema has a description
        orig_desc = desc_path.find(obj2)
        if len(orig_desc) == 0:
            continue

        # Get the tokenized version of each description
        descs = [desc_path.find(obj1)[0].value, desc_path.find(obj2)[0].value]
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
    # Load both objects
    obj1 = json.load(open(sys.argv[1]))
    obj2 = json.load(open(sys.argv[2]))

    # Print similarity
    print(compare_objects(obj1, obj2))
