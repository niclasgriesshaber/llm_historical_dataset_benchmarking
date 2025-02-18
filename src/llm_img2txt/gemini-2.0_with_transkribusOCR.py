#!/usr/bin/env python3
"""
Gemini-2.0 Post-Correction Pipeline (reading existing Transkribus OCR text).

This script:
  1) Ensures each page of the PDF has a PNG image in data/page_by_page/PNG/<pdf_stem>.
     - If not found, it converts the PDF into per-page PNG images.
  2) For each page:
     - Loads existing OCR text from:
       results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
     - Calls Gemini-2.0 (with up to 1-hour retry) passing:
       - The page image (PNG).
       - The existing OCR text, as context in a "post-correction" prompt.
     - Stores the Gemini output as page_000N.txt in a new run folder.
  3) Merges all per-page Gemini outputs into a single TXT file (<pdf_stem>.txt).
  4) Logs usage tokens (prompt/candidate) per page, accumulates them, saves a JSON run log.
  5) If Gemini fails on any page, the script stops immediately.

Example usage:
  ./Gemini-2.0-with-transkribus.py --pdf type-1.pdf [--temperature 0.0]
"""

import argparse
import json
import logging
import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# Gemini
import google.genai as genai
from google.genai import types

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_img2txt"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_img2txt"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# We store PNG in data/page_by_page/PNG/<pdf_stem>

# Transkribus OCR text is pre-existing in:
# results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt

###############################################################################
# Load environment variables (for Gemini)
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)

# Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")  # Must match your .env key

###############################################################################
# Gemini Model Config
###############################################################################
MODEL_NAME = "gemini-2.0-with-transkribus"  # Folder name for final outputs
FULL_MODEL_NAME = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 8192
SEED = 42
RETRY_LIMIT_SECONDS = 3600  # 1 hour max retry for Gemini

###############################################################################
# Utility: Time Formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a number of seconds into H:MM:SS for clean logging.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

###############################################################################
# Argument Parsing
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments: --pdf and --temperature
    """
    parser = argparse.ArgumentParser(
        description="Gemini-2.0 post-correction pipeline (using existing Transkribus OCR)."
    )

    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF file in data/pdfs/, e.g. example.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the Gemini LLM call (default: 0.0)"
    )

    return parser.parse_args()

###############################################################################
# Utility: Find existing 'run_XX' directories to auto-increment
###############################################################################
def find_existing_runs_in_temperature_folder(temp_folder: Path) -> List[int]:
    """
    Look for existing 'run_XX' directories in the temperature-specific folder.
    Returns a list of run numbers (integers).
    """
    if not temp_folder.is_dir():
        return []
    runs = []
    for child in temp_folder.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                run_num = int(child.name.split("_")[1])
                runs.append(run_num)
            except ValueError:
                pass
    return runs

###############################################################################
# Utility: Write a JSON log file in logs/llm_img2txt/<model_name>/
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in the logs/llm_img2txt/<model_name>/ folder.
    """
    pipeline_logs_dir = LOGS_DIR / model_name
    pipeline_logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp_str}.json"
    log_path = pipeline_logs_dir / log_filename

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_dict, f, indent=4)

    logging.info(f"JSON log saved at: {log_path}")

###############################################################################
# Load the Gemini prompt from file
###############################################################################
def load_gemini_prompt() -> str:
    """
    Reads the core Gemini prompt from src/prompts/llm_img2txt/gemini-2.0.txt
    and returns it as a string.
    """
    prompt_path = PROJECT_ROOT / "src" / "prompts" / "llm_img2txt" / "gemini-2.0.txt"
    if not prompt_path.is_file():
        logging.error(f"Prompt file not found at {prompt_path}")
        sys.exit(1)
    return prompt_path.read_text(encoding='utf-8')

###############################################################################
# Multimodal OCR correction prompt function
###############################################################################
def multimodal_ocr_correction(ocr_text: str, image_path: Path) -> str:
    """
    Applies multimodal OCR post-correction while enforcing strict transcription rules.
    
    :param ocr_text: Raw text extracted from OCR (Transkribus output)
    :param image_path: Path to the corresponding image for verification (not used directly in prompt text)
    :return: Corrected and strictly compliant OCR text prompt
    """
    # Load the base prompt from the text file
    base_prompt = load_gemini_prompt().strip()

    # Merge the base prompt and the OCR text
    gemini_prompt_text = (
        f"{base_prompt}\n\n"
        "Below is the OCR output from Transkribus so you know how to spell the archaic words. Please use this information to correct any errors and ensure the text is fully compliant with the strict transcription rules.\n"
        "-- OCR Output (Transkribus) --\n"
        f"{ocr_text}\n"
    )

    return gemini_prompt_text

###############################################################################
# Main Pipeline
###############################################################################
def main() -> None:
    """
    1) Convert PDF => PNG (if not already present) in data/page_by_page/PNG/<pdf_stem>.
    2) For each page:
       - Load pre-existing OCR text from:
         results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
       - Call Gemini => post-correct using the OCR text + image (1-hour max retry).
    3) Write per-page results and a merged final text file.
    4) Write JSON usage log.
    5) Stop if any step fails on any page.
    """
    # -------------------------------------------------------------------------
    # 1. Parse arguments & configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf
    temperature = args.temperature

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== Gemini-2.0 Post-Correction Pipeline ===")
    logging.info(f"PDF: {pdf_name}")
    logging.info(f"Temperature: {temperature} (default=0.0)")

    # Check environment for Gemini
    if not API_KEY:
        logging.error("GOOGLE_API_KEY not set in .env or environment.")
        sys.exit(1)

    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # 2. Prepare PDF path & create PNG if needed
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at {pdf_path}")
        sys.exit(1)

    # PNG folder
    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"Converting PDF -> PNG in {png_dir}")
        png_dir.mkdir(parents=True, exist_ok=True)

        png_pages = convert_from_path(str(pdf_path))
        for i, page_img in enumerate(png_pages, start=1):
            img_path = png_dir / f"page_{i:04d}.png"
            page_img.save(img_path, "PNG")
        logging.info(f"Created {len(png_pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"PNG folder already exists: {png_dir}")

    # Gather page list
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error("No PNG page images found. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)
    logging.info(f"Total pages to process: {total_pages}")

    # -------------------------------------------------------------------------
    # 3. Create results folder => results/llm_img2txt/gemini-2.0-with-transkribus/<pdf_stem>/temperature_X/run_XX/page_by_page
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / MODEL_NAME / pdf_stem
    temp_dir = base_results_path / f"temperature_{temperature}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = find_existing_runs_in_temperature_folder(temp_dir)
    next_run_number = (max(existing_runs) + 1) if existing_runs else 1

    run_dir = temp_dir / f"run_{str(next_run_number).zfill(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # 4. Initialize counters for Gemini usage
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_tokens = 0

    # -------------------------------------------------------------------------
    # 5. Main Loop: For each page, read existing OCR -> Gemini
    # -------------------------------------------------------------------------
    page_text_files = []

    # This is the root folder for pre-existing Transkribus OCR text
    transkribus_ocr_root = PROJECT_ROOT / "results" / "ocr_img2txt" / "transkribus" / pdf_stem / "run_01" / "page_by_page"

    for idx, png_path in enumerate(png_files):
        page_num = idx + 1

        logging.info(f"=== Page {page_num} of {total_pages} ===")
        logging.info(f"PNG: {png_path.name}")

        # 5a) Read the existing OCR text
        transkribus_txt_path = transkribus_ocr_root / f"page_{page_num:04d}.txt"
        if not transkribus_txt_path.is_file():
            logging.error(f"OCR text file not found: {transkribus_txt_path}")
            sys.exit(1)

        ocr_text = transkribus_txt_path.read_text(encoding='utf-8').strip()
        logging.info(f"Loaded OCR text (~{len(ocr_text)} chars).")

        # 5b) Construct the prompt using the new function: includes the strict rules from file
        gemini_prompt_text = multimodal_ocr_correction(ocr_text, png_path)

        gemini_output = None
        page_usage_prompt = 0
        page_usage_candidate = 0
        page_usage_total = 0

        logging.info("Calling Gemini for post-correction...")

        start_retry = time.time()
        while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
            tmp_file = None
            try:
                # Save PNG to a temp file for uploading (Gemini library requires a path)
                with Image.open(png_path) as pil_img, \
                     tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_file = tmp.name
                    pil_img.save(tmp_file, "PNG")

                # Upload the image to the GenAI client
                client = genai.Client(api_key=API_KEY)
                file_upload = client.files.upload(path=tmp_file)

                # The request content consists of two parts:
                #   1) The image
                #   2) The prompt text
                response = client.models.generate_content(
                    model=FULL_MODEL_NAME,
                    contents=[
                        types.Part.from_uri(
                            file_uri=file_upload.uri,
                            mime_type=file_upload.mime_type
                        ),
                        gemini_prompt_text
                    ],
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        seed=SEED
                    ),
                )

                if not response or not response.text:
                    logging.warning("Gemini returned empty response; retrying...")
                    continue

                gemini_output = response.text
                usage = response.usage_metadata
                if usage:
                    page_usage_prompt = usage.prompt_token_count or 0
                    page_usage_candidate = usage.candidates_token_count or 0
                    page_usage_total = usage.total_token_count or (
                        page_usage_prompt + page_usage_candidate
                    )

                # Update global usage
                total_prompt_tokens += page_usage_prompt
                total_candidate_tokens += page_usage_candidate
                total_tokens += page_usage_total

            except Exception as e:
                logging.warning(f"Gemini call failed: {e}. Retrying...")
                continue
            finally:
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass

            # If we have valid output, break the retry loop
            if gemini_output:
                break
        else:
            # Exceeded 1 hour of retries => fail the pipeline
            logging.error(f"Gemini-2.0 failed for page {page_num} after 1 hour. Stopping.")
            sys.exit(1)

        # 5c) Save the Gemini output to page_X.txt
        page_text_path = run_page_dir / f"page_{page_num:04d}.txt"
        with open(page_text_path, 'w', encoding='utf-8') as f:
            f.write(gemini_output)

        page_text_files.append(page_text_path)

        # Log usage
        logging.info(
            f"Gemini usage for page {page_num}: prompt={page_usage_prompt}, "
            f"candidate={page_usage_candidate}, total={page_usage_total}"
        )
        logging.info(
            f"Accumulated usage so far: prompt={total_prompt_tokens}, "
            f"candidate={total_candidate_tokens}, total={total_tokens}"
        )

        # Timing / estimation
        elapsed = time.time() - overall_start_time
        pages_done = page_num
        pages_left = total_pages - pages_done
        avg_time_per_page = elapsed / pages_done
        estimated_total = avg_time_per_page * total_pages
        estimated_remaining = avg_time_per_page * pages_left

        logging.info(
            f"Time so far: {format_duration(elapsed)} | "
            f"Estimated total: {format_duration(estimated_total)} | "
            f"Estimated remaining: {format_duration(estimated_remaining)}"
        )
        logging.info("")

    logging.info("All pages processed successfully. Gemini outputs saved.")

    # -------------------------------------------------------------------------
    # 6. Merge all page text files -> final <pdf_stem>.txt
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path}")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            text_content = txt_file.read_text(encoding='utf-8').strip()
            outf.write(text_content + "\n\n")

    # -------------------------------------------------------------------------
    # 7. Write JSON Log
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start_time
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature": temperature,
        "seed": SEED,
        "pages_count": total_pages,
        "pages_successfully_processed": total_pages,
        "final_text_file": str(final_txt_path),
        "run_directory": str(run_dir),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidate_tokens": total_candidate_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }
    write_json_log(log_info, MODEL_NAME)

    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Prompt tokens:    {total_prompt_tokens}")
    logging.info(f"Candidate tokens: {total_candidate_tokens}")
    logging.info(f"Total tokens:     {total_tokens}")
    logging.info(
        f"Completed in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()