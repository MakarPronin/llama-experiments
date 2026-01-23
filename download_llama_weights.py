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

import torch
from safetensors.torch import load_file
from huggingface_hub import login, hf_hub_download

def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
    
    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left


def load_weights_into_llama(model, param_config, weights_file_paths=[]):
    params = {}
    for file in weights_file_paths:
        current_weights = load_file(file)
        params.update(current_weights)

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):

        # Load attention weights
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")

def download_weights(llama32_config, dir=".", alternative=False):
    llama_size_str = "1B" if llama32_config["emb_dim"] == 2048 else "3B"

    if alternative:
        repo_id = f"huihui-ai/Llama-3.2-{llama_size_str}-Instruct-abliterated"
    else:
        login()
        repo_id = f"meta-llama/Llama-3.2-{llama_size_str}-Instruct"

    if llama_size_str == "1B":
        hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=dir
        )


    else:
        for i in range(1, 3):
            hf_hub_download(
                repo_id=repo_id,
                filename=f"model-0000{i}-of-00002.safetensors",
                local_dir=dir
            )