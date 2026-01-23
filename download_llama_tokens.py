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

from huggingface_hub import login, hf_hub_download

def download_tokens(llama32_config, dir=".", alternative=False):

    if alternative:
        repo_id = "rasbt/llama-3.2-from-scratch"
        filename = "tokenizer.model"
    else:
        login()
        llama_size_str = "1B" if llama32_config["emb_dim"] == 2048 else "3B"
        
        repo_id = f"meta-llama/Llama-3.2-{llama_size_str}-Instruct"
        filename = "original/tokenizer.model"

    tokenizer_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dir
    )

    return tokenizer_file_path