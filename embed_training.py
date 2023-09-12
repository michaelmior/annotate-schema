from abc import ABC, abstractmethod
import argparse
import json
import linecache
import random
from typing import Dict, List, Union

import torch
import tqdm
import torchmetrics
from transformers import AutoModel, AutoTokenizer
import wandb


ALL_KEYWORDS = [
    "exclusiveMinimum",
    "exclusiveMaximum",
    "format",
    "maximum",
    "maxItems",
    "maxLength",
    "maxProperties",
    "minimum",
    "minItems",
    "minLength",
    "minProperties",
    "multipleOf",
    "pattern",
    "uniqueItems",
]


# Code adapted from https://github.com/bigcode-project/bigcode-encoder
class BaseEncoder(torch.nn.Module, ABC):
    def __init__(self, device, max_input_len, maximum_token_len, model_name):
        super().__init__()

        self.device = device
        self.model_name = model_name
        self.tokenizer = self.prepare_tokenizer()
        self.encoder = (
            AutoModel.from_pretrained(model_name, use_auth_token=True)
            .to(self.device)
            .eval()
        )
        self.device = device
        self.max_input_len = max_input_len
        self.maximum_token_len = maximum_token_len

    def prepare_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_auth_token=True
            )

        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.add_special_tokens({"sep_token": "<sep>"})
        tokenizer.add_special_tokens({"cls_token": "<cls>"})
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
        return tokenizer

    @abstractmethod
    def forward(
        self,
    ):
        pass

    def truncate_sentences(self, sentence_list: List[str]) -> List[str]:
        """Truncates list of sentences to a maximum length.

        Args:
            sentence_list (List[str]): List of sentences to be truncated.
            maximum_length (Union[int, float]): Maximum length of any output sentence.

        Returns:
            List[str]: List of truncated sentences.
        """

        truncated_sentences = []

        for sentence in sentence_list:
            truncated_sentences.append(sentence[: self.max_input_len])

        return truncated_sentences

    def encode(self, input_sentences, batch_size=32, **kwargs):
        truncated_input_sentences = self.truncate_sentences(input_sentences)

        n_batches = len(truncated_input_sentences) // batch_size + int(
            len(truncated_input_sentences) % batch_size > 0
        )

        embedding_batch_list = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(truncated_input_sentences))

            with torch.no_grad():
                embedding_batch_list.append(
                    self.forward(truncated_input_sentences[start_idx:end_idx]).detach()
                )

        input_sentences_embedding = torch.cat(embedding_batch_list)

        return [emb.squeeze() for emb in input_sentences_embedding]


class StarEncoder(BaseEncoder):
    def __init__(self, device, max_input_len, maximum_token_len):
        super().__init__(
            device, max_input_len, maximum_token_len, model_name="bigcode/starencoder"
        )

    def set_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_data = {}
        for k, v in inputs.items():
            output_data[k] = v.to(self.device)

        return output_data

    def pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pools a batch of vector sequences into a batch of vector global
        representations. It does so by taking the last vector in the sequence,
        as indicated by the mask.

        Args:
            x (torch.Tensor): Batch of vector sequences with shape [B, T, F].
            mask (torch.Tensor): Batch of masks with shape [B, T].

        Returns:
            torch.Tensor: Pooled version of the input batch with shape [B, F].
        """

        eos_idx = mask.sum(1) - 1
        batch_idx = torch.arange(len(eos_idx), device=x.device)

        mu = x[batch_idx, eos_idx, :]

        return mu

    def pool_and_normalize(
        self,
        features_sequence: torch.Tensor,
        attention_masks: torch.Tensor,
        return_norms: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Temporal pooling of sequences of vectors and projection onto the unit sphere.

        Args:
            features_sequence (torch.Tensor): Inpute features with shape [B, T, F].
            attention_masks (torch.Tensor): Pooling masks with shape [B, T, F].
            return_norms (bool, optional): Whether to additionally return
                                           the norms. Defaults to False.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Pooled and normalized vectors
                                                     with shape [B, F].
        """

        pooled_embeddings = self.pooling(features_sequence, attention_masks)
        embedding_norms = pooled_embeddings.norm(dim=1)

        normalizing_factor = torch.where(  # Only normalize embeddings with norm > 1.0.
            embedding_norms > 1.0, embedding_norms, torch.ones_like(embedding_norms)
        )

        pooled_normalized_embeddings = pooled_embeddings / normalizing_factor[:, None]

        if return_norms:
            return pooled_normalized_embeddings, embedding_norms
        else:
            return pooled_normalized_embeddings

    def forward(self, input_sentences):
        inputs = self.tokenizer(
            [f"<cls>{sentence}<sep>" for sentence in input_sentences],
            padding="longest",
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.encoder(**self.set_device(inputs))
        embedding = self.pool_and_normalize(
            outputs.hidden_states[-1], inputs.attention_mask
        )

        return embedding


class FileDataset(torch.utils.data.Dataset):
    def __init__(self, config, encoder, filename):
        self.config = config
        self.encoder = encoder
        self.filename = filename

        # Count the number of lines in the file
        with open(filename) as f:
            self.max_lines = 0
            for line in f:
                self.max_lines += 1

    def __len__(self):
        return self.max_lines

    def __getitem__(self, idx):
        # Load the line of JSON from the file and
        # serialize the object with indentation
        line = linecache.getline(self.filename, idx + 1)
        obj = json.loads(line)
        tok_str = json.dumps(obj["obj"], indent=4)

        if len(tok_str) <= 1024:
            # Get the embedding of the serialized object
            embedding = self.encoder([tok_str])[0]

            # Get the one-hot encoding of the keyword
            keyword = torch.nn.functional.one_hot(
                torch.tensor(ALL_KEYWORDS.index(obj["keyword"])),
                num_classes=len(ALL_KEYWORDS),
            ).to("cuda:0")

            # Combine the embedding and encodd keyword
            input_tensor = torch.cat([embedding, keyword]).to("cuda:0")

            # Generate the output tensor with label smoothing
            include = self.config["smoothing_epsilon"] * random.random()
            if not obj["is_neg"]:
                include = 1.0 - include
            include = torch.tensor(include).to("cuda:0")

            return input_tensor, include


class TinyModel(torch.nn.Module):
    def __init__(self, config):
        super(TinyModel, self).__init__()

        # Input size is embedding dimension (768)
        # plus one-hot encoding of keywords
        input_size = 768 + len(ALL_KEYWORDS)
        self.linear1 = torch.nn.Linear(input_size, config["hidden_layer_size"])
        self.activation1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(config["dropout_rate"])
        self.linear2 = torch.nn.Linear(config["hidden_layer_size"], 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


def calc_val_stats(model, val_data, batch_size, loss_fn, accuracy_fn):
    model.eval()

    # Load validation data in a single batch
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    loss = 0
    accuracy = 0
    batches = 0
    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            # Round off to undo smoothing
            y_batch = torch.round(y_batch)

            y_pred = model(X_batch).squeeze(1)
            loss += loss_fn(y_pred, y_batch).item()
            accuracy += accuracy_fn(y_pred, y_batch).item()
            batches += 1

    model.train()
    return (loss / batches, accuracy / batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data")
    parser.add_argument("validation_data", nargs="?")
    parser.add_argument("-l", "--learning-rate", default=0.001, type=float)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-d", "--dropout-rate", default=0.2, type=float)
    parser.add_argument("--hidden-layer-size", default=400, type=int)
    parser.add_argument("-s", "--smoothing-epsilon", default=0, type=float)
    parser.add_argument("-b", "--batch-size", default=256, type=int)
    parser.add_argument("--split-seed", default=42, type=int)
    parser.add_argument("-o", "--output-file", default="model.pt")
    parser.add_argument("-t", "--training-split", default=1.0, type=float)

    args = parser.parse_args()

    config = {
        "learning_rate": args.learning_rate,
        "n_epochs": args.num_epochs,
        "dropout_rate": args.dropout_rate,
        "hidden_layer_size": args.hidden_layer_size,
        "smoothing_epsilon": args.smoothing_epsilon,
        "batch_size": args.batch_size,
        "split_seed": args.split_seed,
    }

    # Build an encoder for generating the input
    starencoder = StarEncoder("cuda:0", 10000, 1024)

    # Prepare to load the dataset
    data = FileDataset(config, starencoder, args.training_data)

    # Optionally perform train/test split
    if args.training_split < 0 or args.training_split > 1:
        parser.error("Invalid training split")
    elif args.training_split != 1.0:
        split_generator = torch.Generator().manual_seed(config["split_seed"])
        train_data, val_data = torch.utils.data.random_split(
            data,
            [args.training_split, 1 - args.training_split],
            generator=split_generator,
        )
    else:
        train_data = data
        if args.validation_data:
            val_data = FileDataset(config, starencoder, args.validation_data)
        else:
            val_data = None

    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True
    )

    tinymodel = TinyModel(config).to("cuda:0")

    # Initialize the wandb experiment
    wandb.init(
        project="embed-training",
        config=config,
    )
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("acc", summary="max")
    wandb.define_metric("val_loss", summary="min")
    wandb.define_metric("val_acc", summary="max")
    wandb.watch(tinymodel, log_freq=50)

    loss_fn = torch.nn.BCELoss()
    accuracy_fn = torchmetrics.Accuracy(task="binary").to("cuda:0")
    optimizer = torch.optim.AdamW(tinymodel.parameters(), lr=config["learning_rate"])
    tinymodel.train()
    for epoch in tqdm.tqdm(range(config["n_epochs"]), desc="Epoch", position=0):
        pbar = tqdm.tqdm(dataloader, desc="Batch", position=1, leave=False)
        for batch_num, (X_batch, y_batch) in enumerate(pbar):
            y_pred = tinymodel(X_batch).squeeze(1)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate the accuracy (rounding to undo smoothing)
            accuracy = accuracy_fn(y_pred, torch.round(y_batch))

            pbar.set_postfix(loss=loss.item(), acc=accuracy.item())

            # Periodically log loss and accuracy
            if batch_num % 10 == 0:
                wandb.log({"loss": loss.item(), "acc": accuracy.item()})

        # Calculate validation loss and accuracy
        if val_data:
            (val_loss, val_acc) = calc_val_stats(
                tinymodel, val_data, config["batch_size"], loss_fn, accuracy_fn
            )
            wandb.log({"val_loss": val_loss, "val_acc": val_acc})
            print(f"\nValidation loss: {val_loss:.4f}, accuracy {val_acc:.4f}")

    torch.save(tinymodel.state_dict(), args.output_file)
    wandb.save(args.output_file)


if __name__ == "__main__":
    main()
