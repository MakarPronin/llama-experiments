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
import time
import torch

from utils import get_device, load_checkpoint
from model_stats import get_model_stats
from llama3_model import Llama3Model, LLAMA32_CONFIG
from llama_tokenization import Tokenizer, text_to_token_ids, chat_to_text, token_ids_to_text


def generate_tokens(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def generate_text_simple(model, llama32_config, tokenizer, device, start_context, max_new_tokens=150, top_k=1, temperature=0, use_chat_format=False):
    """
    start_context is either a text (use_chat_format=False)
    or a list of json messages (use_chat_format=True)
    """
    
    if (use_chat_format):
        formatted_text = chat_to_text(start_context, prep_for_generation=True, eos=False)
    else:
        formatted_text = start_context
    
    context_size = llama32_config["context_length"]
    encoded = text_to_token_ids(formatted_text, tokenizer).to(device)

    # Store the length of the prompt so we know where the model starts writing
    prompt_length = encoded.shape[1]

    with torch.no_grad():
        token_ids = generate_tokens(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            top_k=top_k,
            temperature=temperature
        )
    
    if (use_chat_format):
        new_tokens = token_ids[0, prompt_length:] # Shape: [num_new_tokens]

        stop_index = len(new_tokens)
        for i, token in enumerate(new_tokens):
            if token.item() == tokenizer.special["<|eot_id|>"]:
                stop_index = i
                break
        
        final_tokens = new_tokens[:stop_index]
    else:
        final_tokens = token_ids
    
    return token_ids_to_text(final_tokens, tokenizer)


def chat_loop(
        default_system="You are a helpful assistant.",
        llama32_config=LLAMA32_CONFIG,
        checkpoint_file_path = "model_checkpoint.pth",
        tokenizer_file_path="tokenizer.model",
        max_new_tokens=150,
        top_k=1,
        temperature=0,
        use_chat_format=True,
        use_compile=True,
        seed=123):
    
    torch.manual_seed(seed)
    tokenizer = Tokenizer(tokenizer_file_path)
    device = get_device(True)
    if (use_chat_format):
        start_context = [{"role": "system", "content": default_system}]
    else:
        start_context = ""

    model = Llama3Model(llama32_config)
    model.to(device)

    # Compile (Safe Block)
    if use_compile:
        try:
            print("Compiling model (this may take a minute)...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Compile failed or skipped: {e}")

    if os.path.exists(checkpoint_file_path):
        load_checkpoint(model, device, file_path=checkpoint_file_path)
    else:
        print(f"No checkpoint found at {checkpoint_file_path}. Expect meaningless output.")

    get_model_stats(model)

    print("\nYOU CAN START CHATTING WITH THE MODEL\n")

    while True:
        print("YOU:")
        user_content = input()
        print("\nASSISTANT:")
        if (use_chat_format):
            start_context += [{"role": "user", "content": user_content}]
        else:
            start_context += user_content

        assistant_content = generate_text_simple(
            model=model,
            llama32_config=llama32_config,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
            use_chat_format=use_chat_format)
        print(assistant_content, "\n")

        if (use_chat_format):
            start_context += [{"role": "assistant", "content": assistant_content}]
        else:
            start_context += assistant_content