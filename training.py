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

import time
import torch
from pathlib import Path

from model_stats import plot_losses
from llama_tokenization import Tokenizer
from generation import generate_text_simple
from loading_data import get_loaders_finetuning, get_loaders_pretraining
from llama3_model import Llama3Model, LLAMA32_CONFIG
from utils import get_device, read_text_file, read_json_file, print_eta, load_checkpoint, save_checkpoint


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, llama32_config, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter,
                       test_context, # test_context is always a string (not json for chat) since we want to see generated special tokens
                       save_ckpt_freq, tokenizer, all_files, total_files,
                       checkpoint_file_path="model_checkpoint.pth",
                       batch_size=1024, train_ratio=0.90):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()

    try:
        for epoch in range(n_epochs):

            # Iterate over the books in the training corpus
            for index, file_path in enumerate(all_files, 1):
                book_start_time = time.time()
                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                if file_path.endswith(".json"):
                    json_data = read_json_file(file_path)

                    # Initialize new data loaders for each book
                    train_loader, val_loader = get_loaders_finetuning(
                        tokenizer=tokenizer,
                        json_data=json_data,
                        device=device,
                        train_ratio=train_ratio,
                        batch_size=batch_size,
                        allowed_max_length=llama32_config["context_length"],
                        num_workers=0
                    )
                else:
                    text_data = read_text_file(file_path)

                    # Initialize new data loaders for each book
                    train_loader, val_loader = get_loaders_pretraining(
                        tokenizer=tokenizer,
                        text_data=text_data,
                        train_ratio=train_ratio,
                        batch_size=batch_size,
                        max_length=llama32_config["context_length"],
                        stride=llama32_config["context_length"],
                        num_workers=0
                    )
                
                print("Training ...")
                model.train()
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()
                    tokens_seen += input_batch.numel()
                    global_step += 1

                    # Optional evaluation step
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # Generate text passage
                    if global_step % print_sample_iter == 0:
                        model.eval()
                        output_text = generate_text_simple(model, llama32_config, tokenizer, device, test_context)
                        print("Generated sample: ", output_text.replace("\n", " "))  # Compact print format
                        model.train()

                if global_step % save_ckpt_freq == 0:
                    save_checkpoint(model, optimizer, checkpoint_file_path)

                print_eta(start_time, book_start_time, index, total_files)

    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, checkpoint_file_path)

    return train_losses, val_losses, track_tokens_seen


def train(
        data_path="training_data/", #a single txt, a collection of txts, a single json, or a collection of jsons
        checkpoint_file_path="model_checkpoint.pth",
        n_epochs=1,
        print_sample_iter=1000,
        eval_freq=100,
        save_ckpt_freq=100_000,
        learning_rate=5e-4,
        batch_size=4,
        # test_context is always a string since this is a test to see the types of generated tokens, but remember that finetuning often needs special tokens
        test_context="What do llamas eat?",
        llama32_config=LLAMA32_CONFIG,
        tokenizer_file_path="tokenizer.model",
        seed=123
        ):

    device = get_device(True)
    torch.manual_seed(seed)
    model = Llama3Model(llama32_config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    load_checkpoint(model, device, optimizer)
    tokenizer = Tokenizer(tokenizer_file_path) #regular Tokenizer

    data_path = Path(data_path)
    all_files = []

    if data_path.is_file():
        all_files.append(data_path)
    elif data_path.is_dir():
        all_files = [f for f in data_path.iterdir() if f.is_file()]

    total_files = len(all_files)

    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, llama32_config, optimizer, device,
        batch_size=batch_size,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
        eval_iter=1,
        print_sample_iter=print_sample_iter,
        checkpoint_file_path=checkpoint_file_path,
        save_ckpt_freq=save_ckpt_freq,
        test_context=test_context,
        tokenizer=tokenizer, #regular tokenizer
        all_files=all_files, #a single txt, a collection of txts, a single json, or a collection of jsons
        total_files=total_files
    )

    epochs_tensor = torch.linspace(0, n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    save_checkpoint(model, optimizer, checkpoint_file_path)
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")