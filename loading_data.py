# ------------------------------------------------------------------------
# Modified by Makar Pronin (Artificial World) to facilitate experimentation
# with Llama models for non-computer science backgrounds.
#
# The original code and these modifications are licensed under the
# Apache License 2.0.
# ------------------------------------------------------------------------
#
# ORIGINAL COPYRIGHT NOTICE:
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE).
# Source: "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
#   - https://github.com/rasbt/LLMs-from-scratch
# ------------------------------------------------------------------------

from functools import partial
from llama_tokenization import chat_to_token_ids

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, bos=True, eos=True)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(tokenizer, txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

def get_loaders_pretraining(tokenizer, text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        tokenizer,
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloader_v1(
        tokenizer,
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader


class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, roles_to_mask={"system", "user"}, mask_header_for_roles={"assistant"}, ignore_index=-100):
        self.data = data
        self.encoded_items = []
        

        for entry in data:
            input_ids, labels = chat_to_token_ids(
                messages=entry["messages"],
                tokenizer=tokenizer,
                roles_to_mask=roles_to_mask,
                mask_header_for_roles=mask_header_for_roles,
                ignore_index=ignore_index
            )
            
            self.encoded_items.append({
                "input_ids": input_ids,
                "labels": labels
            })

    def __getitem__(self, index):
        return self.encoded_items[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    tokenizer,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    pad_token_id = tokenizer.special["<|finetune_right_pad_id|>"]

    # Determine max length in this batch
    batch_max_length = max(len(item["input_ids"]) + 1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        input_ids = item["input_ids"]
        labels = item["labels"]

        # Lengths of input_ids and labels are the same, so the padding size should be the same too
        pad_len = batch_max_length - len(input_ids)

        # Pad Inputs with pad_token_id
        padded_inputs = input_ids + [pad_token_id] * pad_len
        
        # Pad Labels with ignore_index (-100)
        # Crucial: We do not want to calculate loss on padding tokens
        padded_labels = labels + [ignore_index] * pad_len

        # --- Shifting for Causal Language Modeling ---
        # Input: tokens [0...N-1]
        # Target: tokens [1...N]
        input_tensor = torch.tensor(padded_inputs[:-1])
        target_tensor = torch.tensor(padded_labels[1:])

        # Optional: Truncate to allowed_max_length
        if allowed_max_length is not None:
            input_tensor = input_tensor[:allowed_max_length]
            target_tensor = target_tensor[:allowed_max_length]

        inputs_lst.append(input_tensor)
        targets_lst.append(target_tensor)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

def get_loaders_finetuning(tokenizer, json_data, device, train_ratio, batch_size, allowed_max_length=1024, num_workers=0):
    """
        Expected json_data format: [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, who are you?"},
                    {"role": "assistant", "content": "I am an AI assistant created to help you."},
                    ...
                ]
            },
            {
                "messages": [...]
            },
            ...
        ]
    """
    split_idx = int(train_ratio * len(json_data))

    train_data = json_data[:split_idx]
    val_data = json_data[split_idx:]

    customized_collate_fn = partial(custom_collate_fn, tokenizer=tokenizer, device=device, allowed_max_length=allowed_max_length)

    train_dataset = ChatDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = ChatDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, val_loader

def convert_instructions_to_chat(data, default_system="You are a helpful assistant."):
    """
    Converts a list of Alpaca-style instruction dicts into a Chat format.
    """
    chat_dataset = []

    for entry in data:
        # 1. Create the System message
        messages = [
            {"role": "system", "content": default_system}
        ]

        # 2. Create the User message
        # Combine instruction and input (if it exists)
        user_content = entry["instruction"]
        if entry.get("input"):
            user_content += "\n" + entry["input"]
            
        messages.append({"role": "user", "content": user_content})

        # 3. Create the Assistant message
        messages.append({"role": "assistant", "content": entry["output"]})

        # 4. Wrap in the target structure
        chat_dataset.append({"messages": messages})

    return chat_dataset