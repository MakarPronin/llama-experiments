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



if __name__ == "__main__":
    print('\n\
------------------------------------------------------------------------\n\
Modified by Makar Pronin (Artificial World) to facilitate experimentation\n\
with Llama models for non-computer science backgrounds.\n\
\n\
The original code and these modifications are licensed under the\n\
Apache License 2.0.\n\
------------------------------------------------------------------------\n\
\n\
ORIGINAL COPYRIGHT NOTICE:\n\
Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE).\n\
Source: "Build a Large Language Model From Scratch"\n\
  - https://www.manning.com/books/build-a-large-language-model-from-scratch\n\
  - https://github.com/rasbt/LLMs-from-scratch\n\
------------------------------------------------------------------------\n\
')
    print("\nWELCOME TO THE LOCAL AI EXPERIMENTATION PROJECT!!!\n")



    import sys
    from installing_dependencies import get_validated_input, install_package, install_best_pytorch

    dependencies = ["tqdm", "huggingface_hub", "safetensors", "tiktoken", "torch", "matplotlib", "beautifulsoup4"]
    install_dependencies = get_validated_input(f"DEPENDENCIES: Should dependencies be installed {dependencies}?".replace("]", ")").replace("[", "("), "bool", "y")
    
    if install_dependencies:
        for dependency in dependencies:
            if (dependency == "torch"):
                install_best_pytorch()
            else:
                install_package(dependency)

        input("\nDependencies installed successfully!\nPress Enter to exit and restart the program...")
        sys.exit(0)



    import os
    import json
    import torch
    from training import train
    from generation import chat_loop
    from llama_tokenization import chat_to_text
    from download_llama_tokens import download_tokens
    from llama3_model import LLAMA32_CONFIG, Llama3Model
    from loading_data import convert_instructions_to_chat
    from download_llama_weights import download_weights, load_weights_into_llama
    from utils import download_file, download_gutenberg_books, save_checkpoint

    llama32_config=LLAMA32_CONFIG

    #EDITABLE CONFIGS SECTION
    LLAMA32_CONFIG_CUSTOM = {
        "vocab_size": 128256,            # Vocabulary size
        "context_length": 1024,          # Context length that was used to train the model
        "emb_dim": 1024,                 # Embedding dimension
        "n_heads": 16,                   # Number of attention heads
        "n_layers": 12,                  # Number of layers
        "hidden_dim": 4096,              # Size of the intermediate dimension in FeedForward
        "n_kv_groups": 4,                # Key-Value groups for grouped-query attention
        "rope_base": 500000,             # The base in RoPE's "theta"
        "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
        "rope_freq": {                   # RoPE frequency scaling
            "factor": 1,
            "low_freq_factor": 1,
            "high_freq_factor": 4,
            "original_context_length": 1024,
        }
    }

    GUTENBERG_CONFIG = {
        "delay": 2,
        "languages": ['en'],
        "max_books": 1000,
        "url": "https://www.gutenberg.org/robot/harvest"
    }

    PRETRAINING_CONFIG = {
        "n_epochs": 1,
        "print_sample_iter": 50,
        "eval_freq": 5,
        "save_ckpt_freq": 50,
        "learning_rate": 0.0005,
        "batch_size": 5,
        "test_context": "What do llamas eat?",
        "llama32_config": llama32_config,
        "seed": 123
    }

    FINETUNING_CONFIG = {
        "n_epochs": 1,
        "print_sample_iter": 50,
        "eval_freq": 5,
        "save_ckpt_freq": 50,
        "learning_rate": 0.0005,
        "batch_size": 5,
        "system_context": "You are a helpful assistant.",
        "test_prompt": "What do llamas eat?",
        "llama32_config": llama32_config,
        "seed": 123
    }

    CHAT_CONFIG = {
        "default_system": "You are a helpful assistant.",
        "max_new_tokens": 512,
        "top_k": 1,
        "temperature": 0,
        "use_chat_format": True,
        "seed": 123
    }
    #END OF EDITABLE CONFIGS



    change_config = get_validated_input('MODEL CONFIG: Do you wish to change the default configuration (1B Llama 3.2)? If you change, make sure all downloaded and generated model files are valid. Maybe delete the existing files to provide a clean foundation for the updated model version.', "bool", "n")
    if (change_config):
        
        change_config_to_custom = get_validated_input("MODEL CONFIG: Do you want to change to a custom default configuration in main.py? (Otherwise, you will enter parameters manually now)", "bool", "y") 
        if change_config_to_custom:    
            llama32_config = LLAMA32_CONFIG_CUSTOM
        else:
            available_dtypes = {
                "bfloat16": torch.bfloat16,  # Best for modern GPUs (Ampere+)
                "float16":  torch.float16,   # Standard half-precision
                "float32":  torch.float32,   # Full precision (heavy memory usage)
                "float64":  torch.float64,   # Double precision (rarely used for LLMs)
            }

            llama32_config['vocab_size'] = get_validated_input('MODEL CONFIG: Vocabulary size', "int", llama32_config['vocab_size'])
            llama32_config['context_length'] = get_validated_input('MODEL CONFIG: Context length that was used to train the model', "int", llama32_config['context_length'])
            llama32_config["emb_dim"] = get_validated_input('MODEL CONFIG: Embedding dimension', "int", llama32_config["emb_dim"])
            llama32_config["n_heads"] = get_validated_input('MODEL CONFIG: Number of attention heads', "int", llama32_config["n_heads"])
            llama32_config["n_layers"] = get_validated_input('MODEL CONFIG: Number of layers', "int", llama32_config["n_layers"])
            llama32_config["hidden_dim"] = get_validated_input('MODEL CONFIG: Size of the intermediate dimension in FeedForward', "int", llama32_config["hidden_dim"])
            llama32_config["n_kv_groups"] = get_validated_input('MODEL CONFIG: Key-Value groups for grouped-query attention', "int", llama32_config["n_kv_groups"])
            llama32_config["rope_base"] = get_validated_input('MODEL CONFIG: The base in RoPEs "theta"', "float", llama32_config["rope_base"])
            llama32_config["dtype"] = get_validated_input(
                'MODEL CONFIG: Lower-precision dtype to reduce memory usage', 
                expected_type="choice", 
                default=next(k for k, v in available_dtypes.items() if v == llama32_config["dtype"]),
                options_map=available_dtypes
            )
            llama32_config["rope_freq"]["factor"] = get_validated_input('MODEL CONFIG: RoPE frequency scaling, factor', "float", llama32_config["rope_freq"]["factor"])
            llama32_config["rope_freq"]["low_freq_factor"] = get_validated_input('MODEL CONFIG: RoPE frequency scaling, low frequency factor', "float", llama32_config["rope_freq"]["low_freq_factor"])
            llama32_config["rope_freq"]["high_freq_factor"] = get_validated_input('MODEL CONFIG: RoPE frequency scaling, high frequency factor', "float", llama32_config["rope_freq"]["high_freq_factor"])
            llama32_config["rope_freq"]["original_context_length"] = get_validated_input('MODEL CONFIG: RoPE frequency scaling, original context length', "int", llama32_config["rope_freq"]["original_context_length"])



    should_download_tokens = get_validated_input("TOKENS: Do you wish to download Llama tokens? (This is unnecessary if you already have a suitable file with tokens.)", "bool", "y")
    if should_download_tokens:
        alt_tokens_source = get_validated_input("TOKENS: Do you want to download Llama tokens from an ungated source (https://huggingface.co/rasbt/llama-3.2-from-scratch)?", "bool", "y")

        download_tokens(llama32_config, alternative=alt_tokens_source)



    should_download_weights = get_validated_input("WEIGHTS: Do you want to download standard Llama 3.2 weights? They will not work with a custom model configuration.", "bool", "y")
    if should_download_weights:
        alt_weights_source = get_validated_input("WEIGHTS: Do you want to download Llama weights from an ungated source (https://huggingface.co/huihui-ai/Llama-3.2-1B-Instruct-abliterated)?", "bool", "y")

        download_weights(llama32_config, alternative=alt_tokens_source)



    should_download_gutenberg = get_validated_input("PRETRANING DATA: Would you like to download Gutenberg books? Files will be added to pretraining_data/.", "bool", "n")
    if should_download_gutenberg:

        should_change_default_download = get_validated_input("PRETRANING DATA: Do you want to change the default download parameters?", "bool", "n")
        if should_change_default_download:

            lang_input = get_validated_input("PRETRANING DATA: Languages (comma-separated codes, e.g., en,fr)", "str", ",".join(GUTENBERG_CONFIG["languages"]))
            GUTENBERG_CONFIG["languages"] = [l.strip() for l in lang_input.split(",") if l.strip()]

            GUTENBERG_CONFIG["delay"] = get_validated_input("PRETRANING DATA: Download delay (seconds)", "int", GUTENBERG_CONFIG["delay"])
            GUTENBERG_CONFIG["max_books"] = get_validated_input("PRETRANING DATA: Max number of books", "int", GUTENBERG_CONFIG["max_books"])
            GUTENBERG_CONFIG["url"] = get_validated_input("PRETRANING DATA: Where do you want to download the books from?", "str", "https://www.gutenberg.org/robot/harvest")

        # Unpack dictionary as arguments
        download_gutenberg_books(
            directory="pretraining_data/",
            delay=GUTENBERG_CONFIG["delay"],
            languages=GUTENBERG_CONFIG["languages"],
            max_books=GUTENBERG_CONFIG["max_books"],
            url=GUTENBERG_CONFIG["url"]
        )



    should_download_instructions = get_validated_input("FINETUNING DATA: Do you wish to download chatbot instructions from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json (instructions can be used for fine-tuning)?", "bool", "n")
    if should_download_instructions:
        download_file("instruction-data.json", "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch07/01_main-chapter-code/instruction-data.json")


    should_convert_instructions = get_validated_input("FINETUNING DATA: Do you want to convert instruction-data.json to chat format (chat-data.json)? This is necessary for fine-tuning.", "bool", "n")
    if should_convert_instructions:

        print("Converting the instruction-data.json to chat format...")
        with open("instruction-data.json", "r", encoding="utf-8") as f:
            instructions_json = json.load(f)

        if os.path.exists("chat-data.json"):
            print("File chat-data.json already exists. Skipping.")
        else:
            chat_data_json = convert_instructions_to_chat(instructions_json)

            print('Saving instructions with chat format to chat-data.json...')
            with open("chat-data.json", "w", encoding="utf-8") as f:
                json.dump(chat_data_json, f, indent=4)




    should_load_weights = get_validated_input("LOADING MODEL: Do you want to load weights (model.safetensors or model-00001-of-00002.safetensors, model-00002-of-00002.safetensors) into the model? Weights will be saved to model_checkpoint.pth that is loaded during training and generation.", "bool", "y")
    if should_load_weights:
        llama_size_str = "1B" if llama32_config["emb_dim"] == 2048 else "3B"
        if llama_size_str == "1B":
            weights_file_paths = ["model.safetensors"]
        else:
            weights_file_paths = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

        model = Llama3Model(llama32_config)

        load_weights_into_llama(model, llama32_config, weights_file_paths)

        save_checkpoint(model)



    should_run_pretraining = get_validated_input("PRETRAINING: Do you wish to pretrain the model on dataset in pretraining_data/?", "bool", "n")
    if should_run_pretraining:
        should_change_default_pretraining = get_validated_input("PRETRAINING: Do you wish to change default pretraining settings?", "bool", "n")
        if should_change_default_pretraining:
            PRETRAINING_CONFIG["learning_rate"] = get_validated_input("PRETRAINING: learning rate", "float", PRETRAINING_CONFIG["learning_rate"])
            PRETRAINING_CONFIG["batch_size"] = get_validated_input("PRETRAINING: batch size", "int", PRETRAINING_CONFIG["batch_size"])
            PRETRAINING_CONFIG["n_epochs"] = get_validated_input("PRETRAINING: number of epochs", "int", PRETRAINING_CONFIG["n_epochs"])
            PRETRAINING_CONFIG["save_ckpt_freq"] = get_validated_input("PRETRAINING: save a checkpoint every ? iterations", "int",  PRETRAINING_CONFIG["save_ckpt_freq"])
            PRETRAINING_CONFIG["seed"] = get_validated_input("PRETRAINING: seed", "int", PRETRAINING_CONFIG["seed"])
            PRETRAINING_CONFIG["eval_freq"] = get_validated_input("PRETRAINING: evaluate every ? iterations", "int", PRETRAINING_CONFIG["eval_freq"])
            PRETRAINING_CONFIG["print_sample_iter"] = get_validated_input("PRETRAINING: print a sample every ? iterations", "int", PRETRAINING_CONFIG["print_sample_iter"])
            PRETRAINING_CONFIG["test_context"] = get_validated_input("PRETRAINING: prompt for samples", "str", PRETRAINING_CONFIG["test_context"])

        train(
            data_path="pretraining_data/",
            n_epochs=PRETRAINING_CONFIG["n_epochs"],
            print_sample_iter=PRETRAINING_CONFIG["print_sample_iter"],
            eval_freq=PRETRAINING_CONFIG["eval_freq"],
            save_ckpt_freq=PRETRAINING_CONFIG["save_ckpt_freq"],
            learning_rate=PRETRAINING_CONFIG["learning_rate"],
            batch_size=PRETRAINING_CONFIG["batch_size"],
            test_context=PRETRAINING_CONFIG["test_context"],
            llama32_config=llama32_config,
            seed=PRETRAINING_CONFIG["seed"]
        )



    should_run_finetuning = get_validated_input("FINE-TUNING: Do you wish to fine-tune the model on chat-data.json", "bool", "n")
    if should_run_finetuning:
        should_change_default_finetuning = get_validated_input("FINE-TUNING: Do you wish to change default fine-tuning settings?", "bool", "n")
        if should_change_default_finetuning:
            FINETUNING_CONFIG["learning_rate"] = get_validated_input("FINE-TUNING: learning rate", "float", FINETUNING_CONFIG["learning_rate"])
            FINETUNING_CONFIG["batch_size"] = get_validated_input("FINE-TUNING: batch size", "int", FINETUNING_CONFIG["batch_size"])
            FINETUNING_CONFIG["n_epochs"] = get_validated_input("FINE-TUNING: number of epochs", "int", FINETUNING_CONFIG["n_epochs"])
            FINETUNING_CONFIG["save_ckpt_freq"] = get_validated_input("FINE-TUNING: save a checkpoint every ? iterations", "int",  FINETUNING_CONFIG["save_ckpt_freq"])
            FINETUNING_CONFIG["seed"] = get_validated_input("FINE-TUNING: seed", "int", FINETUNING_CONFIG["seed"])
            FINETUNING_CONFIG["eval_freq"] = get_validated_input("FINE-TUNING: evaluate every ? iterations", "int", FINETUNING_CONFIG["eval_freq"])
            FINETUNING_CONFIG["print_sample_iter"] = get_validated_input("FINE-TUNING: print a sample every ? iterations", "int", FINETUNING_CONFIG["print_sample_iter"])
            FINETUNING_CONFIG["test_prompt"] = get_validated_input("FINE-TUNING: prompt for samples", "str", FINETUNING_CONFIG["test_prompt"])
            FINETUNING_CONFIG["system_context"] = get_validated_input("FINE-TUNING: default system message", "str", FINETUNING_CONFIG["system_context"])

        train(
            data_path="chat-data.json",
            n_epochs=FINETUNING_CONFIG["n_epochs"],
            print_sample_iter=FINETUNING_CONFIG["print_sample_iter"],
            eval_freq=FINETUNING_CONFIG["eval_freq"],
            save_ckpt_freq=FINETUNING_CONFIG["save_ckpt_freq"],
            learning_rate=FINETUNING_CONFIG["learning_rate"],
            batch_size=FINETUNING_CONFIG["batch_size"],
            llama32_config=llama32_config,
            seed=FINETUNING_CONFIG["seed"],
            test_context=chat_to_text(
                [{"role": "system", "content": FINETUNING_CONFIG["system_context"]}, {"role": "user", "content": FINETUNING_CONFIG["test_prompt"]}],
                prep_for_generation=True
            )
        )



    should_chat = get_validated_input("GENERATION: Ready to start chatting with a model?", "bool", "y")
    if should_chat:
        should_change_default_chat = get_validated_input("GENERATION: Would you like to change default chat settings?", "bool", "n")
        if should_change_default_chat:
            use_chat_format = get_validated_input("GENERATION: Use chat format? Using chat format is strongly recommended if the model is fine-tuned for a conversation.", "bool", "y" if CHAT_CONFIG["use_chat_format"] else "n")
            seed = get_validated_input("GENERATION: seed", "int", CHAT_CONFIG["seed"])
            default_system = get_validated_input("GENERATION: default system message", "str", CHAT_CONFIG["default_system"])
            max_new_tokens = get_validated_input("GENERATION: Max new tokens", "int", CHAT_CONFIG["max_new_tokens"])
            top_k = get_validated_input("GENERATION: top k", "int", CHAT_CONFIG["top_k"])
            temperature = get_validated_input("GENERATION: temperature", "int", CHAT_CONFIG["temperature"])

        chat_loop(
            default_system=CHAT_CONFIG["default_system"],
            llama32_config=llama32_config,
            max_new_tokens=CHAT_CONFIG["max_new_tokens"],
            top_k=CHAT_CONFIG["top_k"],
            temperature=CHAT_CONFIG["temperature"],
            use_chat_format=CHAT_CONFIG["use_chat_format"],
            seed=CHAT_CONFIG["seed"]
        )
