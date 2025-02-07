#!/usr/bin/env python3
"""
all_llm_pdf2csv.py

Run one or more LLM pdf-to-csv scripts on one or more PDFs, optionally specifying temperature.
Example usage (from within the scripts folder):
    python all_llm_pdf2csv.py --models gemini-2.0 gpt-4o --pdfs type-1.pdf type-2.pdf type-3.pdf --temperature 0.0

Requirements:
- Expects the individual model scripts to live at ../src/llm_pdf2csv/<model_name>.py
- Expects PDFs to live at ../data/pdfs/<pdf_name>
- If either the script file or the PDF file is missing, that (model, pdf) combination is skipped.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run specified LLM pdf-to-csv scripts on given PDFs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model names (e.g. gemini-2.0, gpt-4o)."
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        required=True,
        help="One or more PDF filenames (e.g. type-1.pdf, type-2.pdf)."
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
    src_dir = os.path.join(PROJECT_ROOT, "src", "llm_pdf2csv")
    pdf_dir = os.path.join(PROJECT_ROOT, "data", "pdfs")

    print("=======================================")
    print("Starting all_llm_pdf2csv.py execution.")
    print("Models to run:", args.models)
    print("PDFs to process:", args.pdfs)
    print("Temperature:", args.temperature)
    print("=======================================")

    for model in args.models:
        # Construct the path to the model script
        model_script_path = os.path.join(src_dir, f"{model}.py")

        # Check if the model script exists
        if not os.path.isfile(model_script_path):
            print(f"[SKIP] Model '{model}' not found at {model_script_path}. Skipping this model.")
            continue

        # For each PDF
        for pdf_name in args.pdfs:
            # Construct the path to the PDF file
            pdf_path = os.path.join(pdf_dir, pdf_name)

            # Check if the PDF file exists
            if not os.path.isfile(pdf_path):
                print(f"[SKIP] PDF '{pdf_name}' not found at {pdf_path}. Skipping this combination.")
                continue

            # Build the command
            cmd = [
                "python",
                model_script_path,
                "--pdf",
                pdf_name,
                "--temperature",
                str(args.temperature),
            ]

            # Print command for logging
            print(f"[RUN] {' '.join(cmd)}")

            # Execute the command
            try:
                subprocess.run(cmd, check=True)
                print(f"[DONE] Finished processing {pdf_name} with model {model}.\n")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Command failed for {pdf_name} with model {model}. Return code: {e.returncode}\n")
                # Continue to the next combination, do not exit the script

    print("=======================================")
    print("all_llm_pdf2csv.py execution finished.")
    print("=======================================")

if __name__ == "__main__":
    main()