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
        

def save_checkpoint(model, optimizer=None, file_path="model_checkpoint.pth"):
    #Saves the model weights and optimizer state.
    print(f"Saving checkpoint to {file_path}...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, file_path)
    print("Saved.")


def load_checkpoint(model, device, optimizer=None, file_path="model_checkpoint.pth"):
    #Loads the model weights and optimizer state.
    if not os.path.exists(file_path):
        print(f"No checkpoint found at {file_path}. Starting from scratch.")
        return

    print(f"Loading checkpoint from {file_path}...")

    # map_location ensures we load onto the correct device (CPU/GPU)
    checkpoint = torch.load(file_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Loaded.")


def download_gutenberg_books(
    directory="pretraining_data/",
    delay=2, 
    languages=['en'],
    max_books=100,
    url="https://www.gutenberg.org/robot/harvest"
):
    # 1. Setup Directory
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    
    # 2. Construct query parameters
    params = [('filetypes[]', 'txt')]
    for lang in languages:
        params.append(('langs[]', lang))
    
    file_links = []
    total_files = len(file_links)
    original_url = url
    prev_url = None
    while len(file_links) < max_books and url is not prev_url:
        if url is not original_url:
            time.sleep(delay)
            harvest_url = url
        else:
            separator = "&" if "?" in url else "?"
            harvest_url = f"{url}{separator}{urlencode(params)}"
        
        # 3. Fetch the harvest page
        print(f"Fetching harvest list from: {harvest_url}")
        response = requests.get(harvest_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]

        new_file_links = []
        prev_url = url
        for link in links:
            if link.endswith(('.zip')):
                new_file_links.append(link)
            else:
                url = link # this is supposed to be a next page link
                
        file_links.extend(new_file_links)
        total_files = len(file_links)
        print(f"Found {min(total_files, max_books)}/{max_books} files.")

    file_links = file_links[:max_books]

    # 4. Download Loop with Extraction
    with tqdm(total=total_files, desc="Overall Progress", unit="book") as pbar:
        for i, file_url in enumerate(file_links):
            filename = file_url.split('/')[-1]
            local_path = os.path.join(directory, filename)
            
            try:
                # A. Download the file
                download_file(local_path, file_url, show_progress=False, verbose=False, verify=False)
                
                # B. Extract if it is a zip file
                if filename.endswith('.zip'):
                    try:
                        with zipfile.ZipFile(local_path, 'r') as zip_ref:
                            # Extract all files to the same directory
                            zip_ref.extractall(directory)
                            
                            # Optional: Log what was extracted (for debugging)
                            # extracted_files = zip_ref.namelist()
                        
                        # C. Cleanup: Remove the .zip file to save space
                        os.remove(local_path)
                        
                    except zipfile.BadZipFile:
                        pbar.write(f"Warning: {filename} was corrupted and could not be unzipped.")

                # Update the master progress bar
                pbar.update(1)
                
                if i < total_files - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                pbar.write(f"Failed to process {filename}: {e}")

