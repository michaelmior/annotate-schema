import argparse
import glob
import os
import random

import peft
import torch
import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

MAX_LENGTH = 512


class JsonDirectoryDataset(torch.utils.data.IterableDataset):
    def __init__(self, dirname, tokenizer):
        # Get the list of files in random order
        self.files = glob.glob(os.path.join(dirname, "*.json"))
        random.shuffle(self.files)

        self.tokenizer = tokenizer
        self.items = None

    def __len__(self):
        # If we have already gone through once, use the calculated length
        if self.items is not None:
            return self.items
        else:
            return super().__len__()

    def __iter__(self):
        items = 0
        for file in self.files:
            data = open(file, "r", encoding="utf-8").read()
            for start_idx in range(0, len(data), MAX_LENGTH):
                items += 1
                yield self.tokenizer(
                    data[start_idx : start_idx + MAX_LENGTH],
                    max_length=MAX_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

        # Remember the number of items
        self.items = items


def get_loader(path, tokenizer, batch_size, collator):
    dataset = JsonDirectoryDataset(path, tokenizer)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=collator,
    )


def calc_val_loss(model, test_data, device):
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        for batch_num, X_batch in enumerate(test_data):
            X_batch = {
                k: v.squeeze(1).to(device)
                for k, v in X_batch.items()
                if k != "token_type_ids"
            }
            outputs = model(**X_batch)
            val_loss += outputs.loss.item()

        val_loss /= batch_num + 1
        return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data")
    parser.add_argument("test_data", nargs="?")
    parser.add_argument("-o", "--output", type=str, default="peft.bin")
    parser.add_argument("-m", "--model", type=str, default="replit/replit-code-v1_5-3b")
    parser.add_argument("-c", "--cpu", default=False, action="store_true")
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("--skip-epochs", default=0, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-4", "--load-in-4bit", default=False, action="store_true")
    parser.add_argument("-8", "--load-in-8bit", default=False, action="store_true")
    parser.add_argument("-a", "--accum-iter", default=1, type=int)
    parser.add_argument("-t", "--target_modules", default="")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-every", default=None, type=int)
    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        parser.error("Only one of --load-in-4bit or --load-in-8bit can be used")

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Construct a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        truncation_side="right",
        paddingside="left",
        return_special_tokens_mask=True,
    )
    tokenizer.model_max_length = 2048
    if args.model.startswith("facebook/incoder-"):
        tokenizer.pad_token = "<pad>"
    else:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # Use the Triton attention implementation if available
    if hasattr(config, "attn_config"):
        config.attn_config["attn_impl"] = "triton"

    kwargs = {}
    if args.load_in_4bit:
        kwargs["load_in_4bit"] = True
    if args.load_in_8bit:
        kwargs["load_in_8bit"] = True

    # Load the model and convert to PEFT using LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, trust_remote_code=True, **kwargs
    )

    # If not using a quantized model, set to the correct device
    if not args.load_in_4bit and not args.load_in_8bit:
        model = model.to(device)

    peft_kwargs = {}
    if args.target_modules:
        peft_kwargs["target_modules"] = args.target_modules.split(",")
    if args.checkpoint:
        model = peft.PeftModel.from_pretrained(model, model_id=args.checkpoint)
    else:
        model.enable_input_require_grads()
        model = peft.get_peft_model(model, peft.LoraConfig(**peft_kwargs))

    # Construct a collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build train and test data loaders
    train_data = get_loader(args.train_data, tokenizer, args.batch_size, data_collator)
    if args.test_data:
        test_data = get_loader(
            args.test_data, tokenizer, args.batch_size, data_collator
        )
    else:
        test_data = None

    optimizer = torch.optim.AdamW(model.parameters())
    pbar = tqdm.tqdm(range(args.skip_epochs, args.num_epochs), desc="Epoch", position=0)
    for epoch in pbar:
        pbar2 = tqdm.tqdm(train_data, desc="Batch", position=1, leave=False)
        model.train()

        stepped = False
        for batch_num, X_batch in enumerate(pbar2):
            X_batch = {
                k: v.squeeze(1).to(device)
                for k, v in X_batch.items()
                if k != "token_type_ids"
            }
            outputs = model(**X_batch)

            # Calculate the loss and backprop
            loss = outputs.loss / args.accum_iter
            loss.backward()

            if batch_num % args.accum_iter == 0:
                optimizer.zero_grad()
                optimizer.step()
                stepped = True

        # Make sure to step the optimizer
        if not stepped:
            optimizer.zero_grad()
            optimizer.step()

        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            model.save_pretrained(
                os.path.normpath(args.output) + "-checkpoint-" + str(epoch)
            )

        if test_data is not None:
            val_loss = calc_val_loss(model, test_data, device)
            pbar.set_postfix(loss=val_loss)

    # Save the final model
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
