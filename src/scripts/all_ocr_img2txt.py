#!/usr/bin/env python3
"""
all_ocr_img2txt.py

Run one or more OCR image-to-text scripts on one or more PDFs
Example usage (from within the scripts folder):
    python all_ocr_img2txt.py --models pytesseractOCR paddleOCR transkribus \
        --pdfs type-1.pdf type-2.pdf

Requirements:
- Expects the individual OCR scripts to live at ../src/ocr_img2txt/<script_name>.py
- Expects PDFs to live at ../data/pdfs/<pdf_name>
- If either the script file or the PDF file is missing, that (ocr_script, pdf) combination is skipped.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run specified OCR scripts on given PDFs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "One or more OCR model/script names. "
            "E.g. pytesseractOCR, paddleOCR, transkribusOCR."
        )
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        required=True,
        help="One or more PDF filenames (e.g. type-1.pdf, type-2.pdf)."
    )

    args = parser.parse_args()

    # Resolve absolute path to this script (so we can build relative paths correctly)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Prepare paths
    src_dir = os.path.join(PROJECT_ROOT, "src", "ocr_img2txt")
    pdf_dir = os.path.join(PROJECT_ROOT, "data", "pdfs")

    print("=======================================")
    print("Starting all_ocr_img2txt.py execution.")
    print("Models (OCR scripts) to run:", args.models)
    print("PDFs to process:", args.pdfs)
    print("=======================================")

    for model in args.models:
        # Construct the path to the model/ocr script
        model_script_path = os.path.join(src_dir, f"{model}.py")

        # Check if the model script exists
        if not os.path.isfile(model_script_path):
            print(f"[SKIP] OCR script '{model}' not found at {model_script_path}. Skipping.")
            continue

        # For each PDF
        for pdf_name in args.pdfs:
            # Construct the path to the PDF file
            pdf_path = os.path.join(pdf_dir, pdf_name)

            # Check if the PDF file exists
            if not os.path.isfile(pdf_path):
                print(f"[SKIP] PDF '{pdf_name}' not found at {pdf_path}. Skipping.")
                continue

            # Build the command
            cmd = [
                "python",
                model_script_path,
                "--pdf",
                pdf_name,
            ]

            # Print command for logging
            print(f"[RUN] {' '.join(cmd)}")

            # Execute the command
            try:
                subprocess.run(cmd, check=True)
                print(f"[DONE] Finished processing '{pdf_name}' with OCR script '{model}'.\n")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Command failed for '{pdf_name}' with OCR script '{model}'. "
                      f"Return code: {e.returncode}\n")
                # Continue to the next combination, do not exit the script

    print("=======================================")
    print("all_ocr_img2txt.py execution finished.")
    print("=======================================")

if __name__ == "__main__":
    main()