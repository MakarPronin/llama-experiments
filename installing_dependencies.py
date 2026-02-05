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

import re
import sys
import platform
import subprocess
import importlib.metadata

def run_cmd(cmd_arr, cwd=None):
    print(f'Attempting to execute "{ " ".join(cmd_arr) }"...') 

    subprocess.check_call(cmd_arr, cwd=cwd)

    print(f'Successfully executed "{ " ".join(cmd_arr) }".')


def install_package(package_name):
    # 1. Check if the package is already installed
    try:
        # This checks the PIP PACKAGE name (e.g., "scikit-learn", not "sklearn")
        importlib.metadata.version(package_name)
        print(f"Package '{package_name}' is already installed. Skipping.")
        return
    except importlib.metadata.PackageNotFoundError:
        pass  # Package not found, proceed to install

    # 2. If not found, install it
    print(f"Attempting to install {package_name}...")
    run_cmd([sys.executable, "-m", "pip", "install", package_name])
    print(f"Successfully installed {package_name}.")


def get_validated_input(question, expected_type="str", default="", options_map=None):    
    """
    Asks a question and checks if the answer matches the expected_type.
    """

    # Define readable names for types
    expected_type_to_expected_str = {
        'int': 'integer',
        'float': 'float',
        'bool': 'y/n',
        'str': 'string',
        'choice': f'one of {list(options_map.keys()) if options_map else ""}'
    }

    # Prepare default suggestion for display
    if expected_type in ["bool", "str", "choice"]:
        default_suggestion = f'"{default}"'
    else:
        default_suggestion = default

    while True:
        prompt = f'\n{question}\nExpected: {expected_type_to_expected_str[expected_type]}\n(Default: {default_suggestion}):\n'
        answer = input(prompt).strip()
        
        # Handle Empty/Default
        if answer == "":
            sys.stdout.write(f"\033[F\033[K{default}\n")
            sys.stdout.flush()
            # If it's a choice map, we need to return the VALUE associated with the default KEY
            if expected_type == 'choice' and default in options_map:
                return options_map[default]
            # For other types, return raw default
            answer = str(default) # Ensure it's a string for processing below
        
        clean_num = answer.replace("_", "")
        
        # 1. Handle BOOLEAN (y/n)
        if expected_type == 'bool':
            clean_answer = answer.lower()
            if clean_answer in ['y', 'yes']:
                return True
            elif clean_answer in ['n', 'no']:
                return False
            else:
                print(f"Invalid input. Please enter 'y' or 'n'. You typed: {answer}")
                continue

        # 2. Handle INTEGER
        elif expected_type == 'int':
            try:
                # First convert to float to handle "5e4" or "123.0"
                val_float = float(clean_num)
                
                # Check if it is actually a whole number
                if val_float.is_integer():
                    return int(val_float)
                else:
                    print(f"The number {answer} is not an integer (it has decimals).")
                    continue
            except ValueError:
                print(f"Invalid integer. Examples: 10, 5e4, 123_456. You typed: {answer}")
                continue

        # 3. Handle FLOAT
        elif expected_type == 'float':
            try:
                # Python's float() handles scientific notation like 5e-4 automatically
                return float(clean_num)
            except ValueError:
                print(f"Invalid number. Examples: 1.5, 5e-4. You typed: {answer}")
                continue

        # 4. Handle CHOICES (Mappings)
        if expected_type == 'choice':
            if answer in options_map:
                return options_map[answer] # Return the OBJECT (e.g., torch.bfloat16)
            else:
                print(f"Invalid choice. Options are: {list(options_map.keys())}")
                continue

        # 5. Handle STRING (Default)
        elif expected_type == 'str':
            if answer: # Ensure it's not empty
                return answer
            else:
                print("Input cannot be empty.")
                continue
        
        else:
            print("Error: Unknown expected_type provided to function.")
            return None
        
def check_nvidia_smi():
    """Checks for NVIDIA GPUs via nvidia-smi."""
    try:
        output = subprocess.check_output(["nvidia-smi"], encoding='utf-8')
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return 0.0

def check_rocm_smi():
    """
    Checks for AMD GPUs via rocm-smi.
    Returns the ROCm version (float) if found (e.g., 6.2), otherwise 0.0.
    """
    # Method 1: Try running rocm-smi (preferred)
    try:
        # Get output from rocm-smi
        output = subprocess.check_output(["rocm-smi"], encoding='utf-8')
        
        # Regex to find "ROCm version: 6.2.1" or similar
        # Output format example: "ROCm version: 6.2"
        match = re.search(r"ROCm version:\s*(\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return 0.0

def get_best_supported_version(version, supported_versions):
    for supported_version in sorted(supported_versions, reverse=True):
        if version >= supported_version:
            return supported_version
        
    return 0.0

def get_os():
    return platform.system()

def install_best_pytorch():
    print("Installing the best available option of PyTorch...")
    os_system = get_os()
    pkgs = ["torch", "torchvision"]
    cmd = [sys.executable, "-m", "pip", "install", *pkgs]

    print(f"Detected OS: {os_system}")

    # --- Scenario A: macOS (Apple Silicon or Intel) ---
    if os_system == "Darwin":
        print("macOS detected. Installing standard PyTorch (MPS/Metal supported)...")
        # Standard pip install works for Mac
        
    # --- Scenario B: Windows or Linux ---
    elif os_system in ["Linux", "Windows"]:
        
        # 1. Check for NVIDIA
        cuda_version = check_nvidia_smi()
        
        # 2. Check for AMD (ROCm)
        # Note: rocm-smi is the tool, but official PyTorch ROCm support is Linux-only.
        rocm_version = check_rocm_smi()

        if cuda_version > 0:
            print(f"NVIDIA GPU detected (CUDA {cuda_version}).")
            torch_cuda_version = get_best_supported_version(cuda_version, [12.6, 12.8, 13.0])
            print(f"Installing PyTorch with CUDA {torch_cuda_version}...")
            cmd += ["--index-url", f"https://download.pytorch.org/whl/cu{int(torch_cuda_version*10)}"]

        elif rocm_version:
            print(f"AMD GPU detected (ROCm {rocm_version}).")
            if os_system == "Linux":
                torch_rocm_version = get_best_supported_version(rocm_version, [6.4])
                print(f"Installing PyTorch with ROCm {torch_rocm_version}...")
                # The official URL for AMD/ROCm support
                cmd += ["--index-url", f"https://download.pytorch.org/whl/rocm{torch_rocm_version}"]
            else:
                print("AMD GPU detected, but official ROCm PyTorch is Linux-only.")
                print("Falling back to standard CPU version. (Use WSL2 for ROCm on Windows).")

        else:
            print("No dedicated GPU detected. Installing standard CPU-only version...")

    else:
        print(f"Unknown OS '{os_system}'. Attempting standard install...")

    # Run the constructed command
    run_cmd(cmd)