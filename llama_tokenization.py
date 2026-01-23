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

import os
import torch
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|finetune_right_pad_id|>": 128004,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({f"<|reserved_{i}|>": 128000 + i
                             for i in range(256)
                             if 128000 + i not in self.special.values()})

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text, allowed_special="all")
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        return self.model.decode(ids)

def header(role):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"

# --- Helper Function to Enforce DRY ---
def segment(role, content="", add_eot=True):
    """Encodes a single segment and appends to both ids and masked_ids."""
    segment = header(role)
    segment += content
    if add_eot:
        segment += "<|eot_id|>"

    return segment

def chat_to_text(messages, prep_for_generation=False, default_sys_message="You are a helpful assistant.", bos=True, eos=True):
    """
        Expected format for messages: [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am an AI assistant created to help you."},
            ...
        ]
        No special tokens are allowed in messages
    """

    # Initialize text
    text = ""

    if bos:
        text += "<|begin_of_text|>"

    # 1. Handle System Message
    # Check if system message exists in the conversation history
    has_system_in_messages = len(messages) > 0 and messages[0]["role"] == "system"

    if not has_system_in_messages:
        text += segment("system", default_sys_message, add_eot=True)

    # 2. Loop through conversation history
    for msg in messages:
        text += segment(msg["role"], msg["content"], add_eot=True)

    # 3. Add Assistant Header for Generation (Inference Mode)
    # If the last message was User, prompt the model to answer.
    if prep_for_generation and messages and messages[-1]["role"] == "user":
        # Note: We do NOT add EOT here, as the model needs to generate the content
        text += segment("assistant", content="", add_eot=False)

    if eos:
        if prep_for_generation:
            print("WARNING: eos=True will be ignored in chat_to_text() because prep_for_generation=True")
        else:
            text += "<|end_of_text|>"
    
    return text


def chat_to_token_ids(messages, tokenizer, bos=True, eos=True, roles_to_mask={"system", "user"}, mask_header_for_roles={"assistant"}, ignore_index=-100):
    input_ids = []
    labels = []

    # 1. Add BOS if necessary
    if bos:
        input_ids.append(tokenizer.special["<|begin_of_text|>"])
        labels.append(ignore_index) # Usually don't predict BOS

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        # Build the segment: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        # Note: We encode the header and content together to ensure correct BPE merging
        segment_text = segment(role, content, add_eot=True)
        segment_ids = tokenizer.encode(segment_text)
        
        input_ids.extend(segment_ids)
        
        # 2. Apply Masking
        if role in roles_to_mask:
            # Mask the entire segment (header + content + EOT)
            labels.extend([ignore_index] * len(segment_ids))
        elif role in mask_header_for_roles:
            # For assistant: Mask the header only, keep the content and EOT
            header_text = header(role)
            header_ids = tokenizer.encode(header_text)
            
            # Label = [IGNORE, IGNORE, ... , TOKEN_ID, TOKEN_ID, ...]
            num_header_tokens = len(header_ids)
            labels.extend([ignore_index] * num_header_tokens)
            labels.extend(segment_ids[num_header_tokens:])
        else:
            labels.extend(segment_ids)

    # 1. Add EOS if necessary
    if eos:
        input_ids.append(tokenizer.special["<|end_of_text|>"])
        labels.append(ignore_index) # Usually don't predict EOS

    return input_ids, labels


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
