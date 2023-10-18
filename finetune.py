import argparse
import glob
import os

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


class JsonDirectoryDataset(torch.utils.data.Dataset):
    def __init__(self, dirname, tokenizer):
        self.files = glob.glob(os.path.join(dirname, "*.json"))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = open(self.files[index], "r", encoding="utf-8").read()
        return self.tokenizer(
            data,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", type=str, default="peft.bin")
    parser.add_argument("-m", "--model", type=str, default="replit/replit-code-v1_5-3b")
    parser.add_argument("-c", "--cpu", default=False, action="store_true")
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Construct a tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        truncation_side="right",
        paddingside="left",
        return_special_tokens_mask=True,
    )
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # Use the Triton attention implementation if available
    if hasattr(config, "attn_config"):
        config.attn_config["attn_impl"] = "triton"

    # Load the model and convert to PEFT using LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, trust_remote_code=True
    ).to(device)
    model = peft.get_peft_model(model, peft.LoraConfig())

    # Construct a collated dataset loader
    dataset = JsonDirectoryDataset(args.input, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(model.parameters())
    for epoch in tqdm.tqdm(range(args.num_epochs), desc="Epoch", position=0):
        pbar = tqdm.tqdm(dataloader, desc="Batch", position=1, leave=False)
        model.train()
        total_loss = 0
        for batch_num, X_batch in enumerate(pbar):
            X_batch = {k: v.squeeze(1).to(device) for k, v in X_batch.items()}
            outputs = model(**X_batch)

            # Calculate the loss and backprop
            loss = outputs.loss
            total_loss += loss.detach().float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save the final model
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()