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
import re
import zlib
import time
import json
import torch
import zipfile
import urllib3
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlencode


def get_device(verbose=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        msg = "Using GPU (CUDA). FASTEST mode enabled."
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        msg = "Using Apple Silicon (MPS). FAST mode enabled."
    else:
        device = torch.device("cpu")
        msg = "Using CPU. This will be SLOW."

    if verbose:
        print(msg)
    
    return device

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
        text_data = file.read()
    return text_data


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors='ignore') as file:
        json_data = json.load(file)
    return json_data


def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    book_end_time = time.time()  # End time of processing this book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_files - index
    average_time_per_book = total_elapsed_time / index
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Training file processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining training files: {eta_h}h {eta_m}m {eta_s}s")


def download_file(file_path, url, show_progress=True, verbose=True, verify=True):
    if os.path.exists(file_path) and verbose:
        # Even if skipping, a quick print helps keep track of loop progress
        print(f"File {file_path} already exists. Skipping.")
        return

    # Use a shorter log if progress bar is hidden to avoid cluttering the console
    if not show_progress and verbose:
        print(f"Downloading {os.path.basename(file_path)}...")

    if not verify and not verbose:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    with requests.get(url, stream=True, timeout=30, verify=verify) as response:
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        mode = response.headers.get('content-encoding', '')
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS) if 'gzip' in mode else None

        # The 'disable' parameter toggles the progress bar visibility
        with tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True, 
            desc=os.path.basename(file_path),
            disable=not show_progress
        ) as progress_bar:
            
            with open(file_path, "wb") as file:
                chunk_size = 1024 * 8 
                for chunk in response.raw.stream(chunk_size, decode_content=False):
                    progress_bar.update(len(chunk))
                    
                    if decompressor:
                        decompressed_chunk = decompressor.decompress(chunk)
                        file.write(decompressed_chunk)
                    else:
                        file.write(chunk)

                if decompressor:
                    file.write(decompressor.flush())

        # Validation
        if total_size != 0 and progress_bar.n != total_size and verbose:
            print(f"ERROR: Download incomplete for {file_path}! Expected {total_size}, got {progress_bar.n}")
        

def save_checkpoint(model, optimizer=None, scheduler=None, epoch=None, file_index=None, global_step=None, 
                    tokens_seen=None, file_path="model_checkpoint.pth"):
    
    if (optimizer is None or epoch is None or file_index is None or global_step is None or 
        tokens_seen is None):
        checkpoint_type = "model_only"
    else:
        checkpoint_type = "training"
    
    checkpoint_state = {
        'type': checkpoint_type,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'file_index': file_index,   # Which file we were working on
        'global_step': global_step, # Total optimization steps
        'tokens_seen': tokens_seen, # For logging
    }

    torch.save(checkpoint_state, file_path)
    print(f"Checkpoint saved!")

def load_checkpoint(model, device, optimizer=None, scheduler=None, file_path="model_checkpoint.pth"):
    if not os.path.exists(file_path):
        return
    
    print(f"Loading checkpoint: {file_path}")
    checkpoint = torch.load(file_path, map_location=device)

    # Load weights
    model.load_state_dict(checkpoint['model'])
    if (optimizer):
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load Scheduler (if it exists in both config and checkpoint)
    if scheduler and checkpoint.get('scheduler'):
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Load Metadata
    checkpoint_type = checkpoint['type']
    start_epoch = checkpoint['epoch']
    start_file_index = checkpoint['file_index']
    global_step = checkpoint['global_step']
    tokens_seen = checkpoint['tokens_seen']
    
    return checkpoint_type, start_epoch, start_file_index, global_step, tokens_seen


def download_gutenberg_books(
    directory="raw_pretraining_data/",
    delay=2,
    max_retries=3,
    languages=['en'],
    max_books=100,
    url="https://www.gutenberg.org/robot/harvest"
):
    # Setup Directory
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    
    # Construct query parameters
    params = [('filetypes[]', 'txt')]
    for lang in languages:
        params.append(('langs[]', lang))
    
    file_links = []
    original_url = url
    prev_url = None
    while len(file_links) < max_books and url is not prev_url:
        if url is not original_url:
            time.sleep(delay)
            harvest_url = url
        else:
            separator = "&" if "?" in url else "?"
            harvest_url = f"{url}{separator}{urlencode(params)}"
        
        # Fetch the harvest pages
        print(f"Fetching harvest list from: {harvest_url}")
        response = requests.get(harvest_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]

        new_file_links = []
        prev_url = url
        for link in links:
            if link.endswith(('.zip', '.txt')):
                new_file_links.append(link)
            else:
                url = link # this is supposed to be a next page link
                
        file_links.extend(new_file_links)
        print(f"Found {min(len(file_links), max_books)}/{max_books} files.")

    file_links = file_links[:max_books]
    total_files = len(file_links)

    # Download Loop with Extraction    
    with tqdm(total=total_files, desc="Overall Progress", unit="book") as pbar:
        for i, file_url in enumerate(file_links):
            filename = file_url.split('/')[-1]
            local_path = os.path.join(directory, filename)
            
            # Retry Logic: Attempt up to 3 times
            for attempt in range(max_retries):
                try:
                    download_file(local_path, file_url, show_progress=False, verbose=False, verify=False)
                    
                    if filename.endswith('.zip'):
                        # Note: We treat BadZipFile as a failure here so it triggers a retry 
                        # (in case the corruption was due to a bad download)
                        with zipfile.ZipFile(local_path, 'r') as zip_ref:
                            zip_ref.extractall(directory)
                        
                        # SUCCESS CLEANUP: Only remove the file if it was a zip we extracted
                        if os.path.exists(local_path):
                            os.remove(local_path)
                    
                    # If we reach this line, everything succeeded.
                    # We keep .txt files, but removed .zips above.
                    break 

                except Exception as e:
                    # FAILURE CLEANUP: Remove partial/corrupt files before retrying
                    if os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except OSError:
                            pbar.write(f"Could not complete a cleanup of {filename} after a failed attempt.")
                    
                    # Log the failure
                    if attempt < max_retries - 1:
                        pbar.write(f"Attempt {attempt + 1} failed for {filename}: {e}. Retrying...")
                        time.sleep(delay)
                    else:
                        # Final failure after max retries
                        pbar.write(f"Failed to process {filename} after {max_retries} attempts: {e}")

            # Update master progress bar (once per file, not per attempt)
            pbar.update(1)
            
            if i < total_files - 1:
                time.sleep(delay)


def strip_gutenberg_headers(text):
    """
    Removes Project Gutenberg headers and footers (modern + legacy).
    """

    start_patterns = [
        r"\*\*\*\s*START OF.*?PROJECT GUTENBERG EBOOK",
        r"\*{3}\s*START\*{2}THE SMALL PRINT!\*{2}FOR PUBLIC DOMAIN ETEXTS\*{2}START\*{3}",
    ]

    end_patterns = [
        r"\*\*\*\s*END OF.*?PROJECT GUTENBERG EBOOK",
        r"\*END\*THE SMALL PRINT!.*?END\*",
    ]

    start_index = 0
    end_index = len(text)

    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            start_index = match.end()
            break

    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            end_index = match.start()
            break

    if end_index < start_index:
        return text

    return text[start_index:end_index].strip()


def combine_files(
    source_dir="raw_pretraining_data/", 
    target_dir="pretraining_data/", 
    train_ratio=0.90, 
    max_size_mb=500,  # Caps BOTH training and validation files
    strip_headers=True, 
    start_separator="<|begin_of_text|>", 
    end_separator="<|end_of_text|>", 
    fallback_encoding="latin1"
):
    """
    Iterates source files, splits them into Train/Val.
    - Train parts are combined into chunks (max_size_mb).
    - Validation parts are combined into chunks (max_size_mb).
    """
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(source_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]

    # --- Buffers ---
    train_buffer = []
    val_buffer = []
    
    # --- Size Trackers ---
    current_train_size = 0
    current_val_size = 0
    
    # --- File Counters ---
    train_file_counter = 1
    val_file_counter = 1

    print(f"Processing {len(all_files)} files.")
    print(f"Split ratio: {train_ratio*100:.1f}% Train / {(1-train_ratio)*100:.1f}% Val")
    print(f"File Size Cap: {max_size_mb} MB")

    for file_path in tqdm(all_files, desc="Splitting & Combining"):
        
        # 1. READ
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        if strip_headers:
            content = strip_gutenberg_headers(content)

        # Normalize newlines
        content = re.sub(r"\n\s*\n", "\n\n", content)

        # 2. SPLIT
        split_idx = int(len(content) * train_ratio)
        train_part = content[:split_idx]
        val_part = content[split_idx:]

        # 3. PROCESS TRAINING PART
        if train_part.strip():
            train_part_size = len(train_part.encode("utf-8"))
            
            if current_train_size + train_part_size > max_size_mb * 1024 * 1024:
                add_buffer_to_file(train_buffer, target_dir, f"train_{train_file_counter}.txt", start_separator, end_separator)
                train_file_counter += 1
                train_buffer = [train_part]
                current_train_size = train_part_size
            else:
                train_buffer.append(train_part)
                current_train_size += train_part_size

        # 4. PROCESS VALIDATION PART
        if val_part.strip():
            val_part_size = len(val_part.encode("utf-8"))
            
            if current_val_size + val_part_size > max_size_mb * 1024 * 1024:
                add_buffer_to_file(val_buffer, target_dir, f"validation_{val_file_counter}.txt", start_separator, end_separator)
                val_file_counter += 1
                val_buffer = [val_part]
                current_val_size = val_part_size
            else:
                val_buffer.append(val_part)
                current_val_size += val_part_size

    # 5. FINAL FLUSH
    if train_buffer:
         add_buffer_to_file(train_buffer, target_dir, f"train_{train_file_counter}.txt", start_separator, end_separator)
    if val_buffer:
         add_buffer_to_file(val_buffer, target_dir, f"validation_{val_file_counter}.txt", start_separator, end_separator)

    print(f"\nDone.")
    print(f" -> Created Training files: {train_file_counter}")
    print(f" -> Created Validation files: {val_file_counter}")
    return train_file_counter, val_file_counter

def add_buffer_to_file(content_list, target_dir, filename, start_sep, end_sep):
    """Helper to write a buffer list to a file"""
    out_path = os.path.join(target_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        for text in content_list:
            if text.strip():
                f.write(f"{start_sep}\n{text}\n{end_sep}\n")


def get_autocast_config(device):
    """
    Probes the hardware to find the best data type.
    Returns the dtype (bfloat16 or float16) and device string.
    """
    # 1. Handle CPU (Standard is bfloat16 for CPU AMP)
    if device.type == 'cpu':
        return 'cpu', torch.bfloat16
        
    # 2. Default to float16 (Best for Turing/Pascal/Volta - e.g., GTX 10xx, 16xx, RTX 20xx)
    dtype = torch.float16
    
    # 3. Check for NATIVE bfloat16 support on CUDA
    if device.type == 'cuda':
        # Get the compute capability (returns tuple like (7, 5))
        major, _ = torch.cuda.get_device_capability(device)
        
        # Only use bfloat16 if architecture is Ampere (8.0) or newer (e.g., RTX 30xx, 40xx, A100)
        # We explicitly check 'major >= 8' to avoid false positives on Turing (7.5).
        if major >= 8 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

    return device.type, dtype