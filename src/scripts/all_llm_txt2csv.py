#!/usr/bin/env python3
"""
all_llm_txt2csv.py

Run one or more LLM text-to-csv scripts on one or more TXT files, optionally specifying temperature.

Example usage (from within the scripts folder):
    python all_llm_txt2csv.py --models gemini-2.0 gpt-4o --txts type-1.txt type-2.txt --temperature 0.0

Requirements:
- Expects the individual model scripts to live at ../llm_txt2csv/<model_name>.py
- Expects TXT files to live at ../../data/ground_truth/txt/<txt_file>
- If either the script file or the TXT file is missing, that (model, txt) combination is skipped.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run specified LLM text-to-csv scripts on given TXT files."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model names (e.g. gemini-2.0, gpt-4o)."
    )
    parser.add_argument(
        "--txts",
        nargs="+",
        required=True,
        help="One or more TXT filenames (e.g. type-1.txt, type-2.txt)."
    )
    parser.add_argument(
        "--temperature",
        default=0.0,
        help="Temperature to pass to the model scripts (default = 0.0)."
    )

    args = parser.parse_args()

    # Resolve absolute path to this script (so we can build relative paths correctly)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Prepare paths
    src_dir = os.path.join(PROJECT_ROOT, "src", "llm_txt2csv")
    txt_dir = os.path.join(PROJECT_ROOT, "data", "ground_truth", "txt")

    print("=======================================")
    print("Starting all_llm_txt2csv.py execution.")
    print("Models to run:", args.models)
    print("TXT files to process:", args.txts)
    print("Temperature:", args.temperature)
    print("=======================================")

    for model in args.models:
        # Construct the path to the model script
        model_script_path = os.path.join(src_dir, f"{model}.py")

        # Check if the model script exists
        if not os.path.isfile(model_script_path):
            print(f"[SKIP] Model '{model}' not found at {model_script_path}. Skipping.")
            continue

        # For each TXT file
        for txt_name in args.txts:
            # Construct the path to the TXT file
            txt_path = os.path.join(txt_dir, txt_name)

            # Check if the TXT file exists
            if not os.path.isfile(txt_path):
                print(f"[SKIP] TXT '{txt_name}' not found at {txt_path}. Skipping.")
                continue

            # Build the command
            cmd = [
                "python",
                model_script_path,
                "--txt",
                txt_name,
            ]

            # Print command for logging
            print(f"[RUN] {' '.join(cmd)}")

            # Execute the command
            try:
                subprocess.run(cmd, check=True)
                print(f"[DONE] Finished processing {txt_name} with model {model}.\n")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Command failed for {txt_name} with model {model}. Return code: {e.returncode}\n")
                # Continue to the next combination, do not exit the script

    print("=======================================")
    print("all_llm_txt2csv.py execution finished.")
    print("=======================================")

if __name__ == "__main__":
    main()