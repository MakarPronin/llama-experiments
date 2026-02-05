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
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from model_stats import plot_losses
from llama_tokenization import Tokenizer
from generation import generate_text_simple
from llama3_model import Llama3Model, LLAMA32_CONFIG
from loading_data import get_loaders_finetuning, get_loaders_pretraining
from utils import get_device, read_text_file, read_json_file, print_eta, load_checkpoint, save_checkpoint, get_autocast_config


def get_loader(file_path, tokenizer, llama32_config, device, batch_size, num_workers):
    if str(file_path).endswith(".json"):
        json_data = read_json_file(file_path)
        train_loader, val_loader = get_loaders_finetuning(
            tokenizer=tokenizer, json_data=json_data, device=device,
            train_ratio=1, batch_size=batch_size,
            allowed_max_length=llama32_config["context_length"],
            num_workers=num_workers
        )
    else:
        text_data = read_text_file(file_path)
        train_loader, val_loader = get_loaders_pretraining(
            tokenizer=tokenizer, text_data=text_data,
            train_ratio=1, batch_size=batch_size,
            max_length=llama32_config["context_length"],
            stride=llama32_config["context_length"],
            num_workers=num_workers
        )

    return train_loader


import math

def calculate_total_steps(all_files, tokenizer, llama32_config, batch_size, accumulation_steps, num_workers, n_epochs):
    """
    Calculates the EXACT total steps by iterating all files and counting batches.
    Includes logic for 'End of Book' steps where remaining gradients are flushed.
    """
    if not all_files:
        return 0
    
    print(f"Calculating exact steps for {len(all_files)} files across {n_epochs} epochs...")
    print("Note: This reads all data once. For large datasets, this may take time.")
    
    total_steps_per_epoch = 0
    
    for i, file_path in enumerate(all_files):
        # We load the data exactly as the training loop does to ensure 1:1 accuracy
        # We use 'cpu' to avoid fragmenting GPU memory before training starts
        train_loader = get_loader(
            file_path, tokenizer, llama32_config, "cpu", batch_size, num_workers
        )
        
        num_batches = len(train_loader)
        
        # LOGIC MATCH: In a training loop, 'accum_track' resets per file.
        # If a file has 9 batches and accumulation is 8:
        #   - 1 step at batch 8
        #   - 1 step at batch 9 (triggered by 'End of Book Step' block)
        # Therefore, steps per file = ceil(batches / accumulation)
        steps_in_file = math.ceil(num_batches / accumulation_steps)
        
        total_steps_per_epoch += steps_in_file
        
        # Print progress every 10 files
        if (i + 1) % 10 == 0:
            print(f" - Scanned {i + 1}/{len(all_files)} files...")

    total_training_steps = total_steps_per_epoch * n_epochs
    
    print(f"Exact Calculation: {total_training_steps} total steps ({total_steps_per_epoch} per epoch).")
    return total_training_steps


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
        
    num_batches_processed = 0
    
    # Use standard loop and break manually to ensure we count exactly what we processed
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
            
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        num_batches_processed += 1
        
    if num_batches_processed == 0:
        return float("nan")
        
    return total_loss / num_batches_processed


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        file_train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return file_train_loss, val_loss


def training_step(model, device, llama32_config, tokenizer, train_loader, val_loader, scaler, optimizer, scheduler, eval_freq,
    eval_iter, test_context, file_train_losses, val_losses, track_tokens_seen, global_step, tokens_seen, epoch, print_sample_iter):

# 1. Unscale before clipping
    if scaler:
        scaler.unscale_(optimizer)
    
    # 2. Gradient Clipping (Crucial for Llama stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 3. Step
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    
    # 4. Scheduler Step
    if scheduler:
        scheduler.step()

    optimizer.zero_grad()
    global_step += 1

    # --- EVALUATION ---
    if global_step % eval_freq == 0:
        tl, vl = evaluate_model(model, train_loader, val_loader, device, eval_iter)
        file_train_losses.append(tl)
        val_losses.append(vl)
        track_tokens_seen.append(tokens_seen)
        print(f"Ep {epoch+1} (Step {global_step}): Train {tl:.3f}, Val {vl:.3f}")

    # --- GENERATION ---
    if global_step % print_sample_iter == 0:
        model.eval()
        print(f"\n[Generating sample at Step {global_step}]")
        out = generate_text_simple(model, llama32_config, tokenizer, device, test_context)
        print(f"Output: {out.replace('\n', ' ')}\n") 
        model.train()

    return global_step


def train_model_simple(model, llama32_config, optimizer, device, n_epochs,
                       eval_freq, eval_iter, ckpt_freq_after_file, print_sample_iter,
                       test_context, tokenizer, all_files, total_files,
                       checkpoint_file_path, batch_size, val_loader,
                       accumulation_steps, scheduler, num_workers, use_autocast,
                       start_epoch=0, start_file_index=0, 
                       start_global_step=0, start_tokens_seen=0):

    file_train_losses, val_losses, track_tokens_seen = [], [], []
    
    # Initialize state from arguments
    tokens_seen = start_tokens_seen
    global_step = start_global_step
    last_checkpoint_step = start_global_step
    
    start_time = time.time()
    amp_device, amp_dtype = get_autocast_config(device)
    use_scaler = use_autocast and (amp_dtype == torch.float16) and (device.type == 'cuda')
    scaler = GradScaler() if use_scaler else None 

    print(f"Training. Epoch: {start_epoch+1}, File Index: {start_file_index}")

    # Loop from start_epoch instead of 0
    for epoch in range(start_epoch, n_epochs):
        
        for index, file_path in enumerate(all_files):
            
            # --- RESUME LOGIC ---
            # If we are in the starting epoch, skip files we already finished
            # Note: We re-process the file we were currently on (start_file_index)
            # to ensure we don't miss data from a partial file run.
            if epoch == start_epoch and index < start_file_index:
                continue 
            # --------------------

            book_start_time = time.time()
            
            train_loader = get_loader(file_path, tokenizer, llama32_config, device, batch_size, num_workers)

            if len(train_loader) == 0: continue

            print(f"Processing file {index+1}/{total_files} ({len(train_loader)} batches): {file_path}.")
            model.train()
            accum_track = 0
            
            for input_batch, target_batch in train_loader:
                # Setup Autocast
                ctx = autocast(device_type=amp_device, dtype=amp_dtype) if use_autocast else nullcontext()

                with ctx:
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss = loss / accumulation_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_track += 1

                if accum_track % accumulation_steps == 0:
                    global_step = training_step(model, device, llama32_config, tokenizer, train_loader, val_loader, scaler, optimizer, scheduler, eval_freq,
                        eval_iter, test_context, file_train_losses, val_losses, track_tokens_seen, global_step, tokens_seen, epoch, print_sample_iter)

                tokens_seen += input_batch.numel()

            # End of Book Step. Removes accumulated gradients.
            if accum_track % accumulation_steps != 0:
                global_step = training_step(model, device, llama32_config, tokenizer, train_loader, val_loader, scaler, optimizer, scheduler, eval_freq,
                    eval_iter, test_context, file_train_losses, val_losses, track_tokens_seen, global_step, tokens_seen, epoch, print_sample_iter)
                
            print_eta(start_time, book_start_time, index, total_files)

            # End of Book Save
            if (index == len(all_files) - 1):
                last_checkpoint_step = global_step
                save_checkpoint(model, optimizer, scheduler, epoch+1, 0, global_step, tokens_seen, checkpoint_file_path)
            elif (global_step - last_checkpoint_step >= ckpt_freq_after_file):
                last_checkpoint_step = global_step
                save_checkpoint(model, optimizer, scheduler, epoch, index+1, global_step, tokens_seen, checkpoint_file_path)

    return file_train_losses, val_losses, track_tokens_seen


def train(
        data_path="training_data/", 
        checkpoint_file_path="model_checkpoint.pth",
        n_epochs=1,
        print_sample_iter=100,
        eval_freq=100,
        ckpt_freq_after_file=100,
        learning_rate=0.0003,
        batch_size=5,
        accumulation_steps=8,
        num_workers=2,
        use_autocast=True,
        use_compile=True,
        use_scheduler=True,
        test_context="The meaning of life is",
        llama32_config=LLAMA32_CONFIG,
        tokenizer_file_path="tokenizer.model",
        seed=123
    ):

    device = get_device(True)
    torch.manual_seed(seed)
    tokenizer = Tokenizer(tokenizer_file_path)

    # File setup
    data_path = Path(data_path)
    all_files = []
    if data_path.is_file():
        all_files.append(str(data_path))
    elif data_path.is_dir():
        all_files = [str(f) for f in data_path.iterdir() if f.is_file()]

    # FILTER: Separate Train and Validation files based on filename
    # Assumes your splitter names them "train_*.txt" and "validation_*.txt"
    train_files = sorted([f for f in all_files if "train_" in f])
    val_files = sorted([f for f in all_files if "validation_" in f])

    if not train_files:
        print("No training files found! (Looking for 'train_*.txt')")
        return

    if not val_files:
        print("Warning: No validation files found. Validation loss will be skipped/NaN.")
        val_loader = [] # Empty loader
    else:
        # LOAD VALIDATION (Just use the first validation file for speed)
        print(f"Loading validation file: {val_files[0]}")
        # We reuse get_loader because it loads 100% of the data, which is what we want for validation file
        val_loader = get_loader(val_files[0], tokenizer, llama32_config, "cpu", batch_size, num_workers)
    
    total_train_files = len(train_files)
    if total_train_files == 0:
        print("No files found!")
        return
    
    # Initialize Model
    model = Llama3Model(llama32_config)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    # --- SCHEDULER SETUP ---
    # We must init the scheduler with TOTAL steps before loading state
    scheduler = None
    if use_scheduler:
        # We now calculate total steps directly (including epochs)
        total_training_steps = calculate_total_steps(
            train_files, tokenizer, llama32_config, batch_size, accumulation_steps, num_workers, n_epochs
        )
        
        # Init scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps)

    # --- LOAD STATE (If Exists) ---
    checkpoint_type = ''
    if not os.path.exists(checkpoint_file_path):
        print(f"No checkpoint found at {checkpoint_file_path}. Starting from scratch.")
    else:    
        checkpoint_type, start_epoch, start_file_index, start_global_step, start_tokens = load_checkpoint(
            model, device, optimizer, scheduler, checkpoint_file_path
        )

    if (checkpoint_type != 'training'):
        print(f"The loaded checkpoint is model_only. Training will be started from epoch 0, file 0, step 0.")
        start_epoch = 0
        start_file_index = 0
        start_global_step = 0
        start_tokens = 0
    
    if use_compile:
        try:
            print("Compiling model (this may take a minute)...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Compile failed or skipped: {e}")

    # --- START TRAINING ---
    file_train_losses, val_losses, tokens_seen = train_model_simple(
        model=model, 
        llama32_config=llama32_config, 
        optimizer=optimizer, 
        device=device,
        n_epochs=n_epochs,
        eval_freq=eval_freq,
        ckpt_freq_after_file=ckpt_freq_after_file,
        eval_iter=5,
        print_sample_iter=print_sample_iter,
        test_context=test_context,
        tokenizer=tokenizer,
        all_files=train_files,
        total_files=total_train_files,
        checkpoint_file_path=checkpoint_file_path,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        scheduler=scheduler,
        val_loader=val_loader,
        num_workers=num_workers,
        use_autocast=use_autocast,
        # RESUME ARGS
        start_epoch=start_epoch,
        start_file_index=start_file_index,
        start_global_step=start_global_step,
        start_tokens_seen=start_tokens,
    )

    # Plotting
    epochs_tensor = torch.linspace(0, n_epochs, len(file_train_losses))
    plot_losses(epochs_tensor, tokens_seen, file_train_losses, val_losses)
    
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")