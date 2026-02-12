# llama-experiments
Code that facilitates experimentation with Llama 3.2 models for non-computer science backgrounds.

# Instructions
Run main.py as an administrator from its folder (/llama_experiments). All further instructions will be provided on the screen.



# Logging Output to File

To run the script and save both the output and errors to a log file (appending to it) while still seeing everything on the console, use the following commands:

### Linux / macOS / Git Bash
python main.py | tee -a output.txt

### Windows PowerShell
python main.py | Tee-Object -FilePath output.txt -Append



# License
Modified by Makar Pronin (Artificial World) to facilitate experimentation
with Llama models for non-computer science backgrounds.

The original code and these modifications are licensed under the
Apache License 2.0.


ORIGINAL COPYRIGHT NOTICE:
Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE).
Source: "Build a Large Language Model From Scratch"
  - https://www.manning.com/books/build-a-large-language-model-from-scratch
  - https://github.com/rasbt/LLMs-from-scratch