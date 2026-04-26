#!/usr/bin/env python3
"""
###############################################################################
# REVISION EXPERIMENT SCRIPT
#
# This script is an adaptation of the Gemini 2.0 Transkribus post-correction
# pipeline for the Gemini 3.1 Pro Preview model. It is part of a revision
# experiment that stores results and logs in separate directories
# (results_revisions/ and logs_revisions/) to keep them isolated from the
# original experiment outputs.
#
# Changes from gemini-2.0_with_transkribusOCR.py:
#   - MODEL_NAME changed from "gemini-2.0-with-transkribus" to "gemini-3.1-with-transkribus"
#   - FULL_MODEL_NAME changed from "gemini-2.0-flash" to "gemini-3.1-pro-preview"
#   - RESULTS_DIR changed from "results/" to "results_revisions/"
#   - LOGS_DIR changed from "logs/" to "logs_revisions/"
#   - Prompt file changed from "gemini-2.0.txt" to "gemini-3.1.txt"
#   - NOTE: Transkribus OCR source remains at "results/ocr_img2txt/transkribus/"
#     (the original results, NOT results_revisions)
#
# All pipeline logic remains identical to the original.
###############################################################################

Gemini-3.1 Post-Correction Pipeline (reading existing Transkribus OCR text).

This script:
  1) Ensures each page of the PDF has a PNG image in data/page_by_page/PNG/<pdf_stem>.
     - If not found, it converts the PDF into per-page PNG images.
  2) For each page:
     - Loads existing OCR text from:
       results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
     - Calls Gemini-3.1 (no retry, exits on failure) passing:
       - The page image (PNG).
       - The existing OCR text, as context in a "post-correction" prompt.
     - Stores the Gemini output as page_000N.txt in a new run folder.
  3) Merges all per-page Gemini outputs into a single TXT file (<pdf_stem>.txt).
  4) Logs usage tokens (prompt/candidate) per page, accumulates them, saves a JSON run log.
  5) If Gemini fails on any page, the script stops immediately.

Example usage:
  ./gemini-3.1_with_transkribusOCR.py --pdf type-1.pdf [--temperature 0.0]
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

# Gemini -- Google GenAI library, used for both Gemini 2.0 and 3.1 models
import google.genai as genai
from google.genai import types

###############################################################################
# Project Paths
###############################################################################
# PROJECT_ROOT is two levels up from this script (src/llm_img2txt/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where source PDFs and page images are stored
DATA_DIR = PROJECT_ROOT / "data"

# REVISION: Changed from "results" to "results_revisions" to isolate revision outputs
# Output directory for the post-corrected text results
RESULTS_DIR = PROJECT_ROOT / "results_revisions" / "llm_img2txt"

# REVISION: Changed from "logs" to "logs_revisions" to isolate revision logs
LOGS_DIR = PROJECT_ROOT / "logs_revisions" / "llm_img2txt"

# Path to the .env file containing API keys
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# We store PNG in data/page_by_page/PNG/<pdf_stem>

# Transkribus OCR text is pre-existing in the ORIGINAL results directory:
# results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
# NOTE: This intentionally reads from "results/" (NOT "results_revisions/") because
# the Transkribus OCR was produced during the original experiment and serves as input.

###############################################################################
# Load environment variables (for Gemini)
###############################################################################
# Load the .env file so we can access GOOGLE_API_KEY
load_dotenv(dotenv_path=ENV_PATH)

# Gemini API key from .env
API_KEY = os.getenv("GOOGLE_API_KEY")  # Must match your .env key

###############################################################################
# Gemini Model Config
###############################################################################
# REVISION: Changed from "gemini-2.0-with-transkribus" to "gemini-3.1-with-transkribus"
# This is used as the subfolder name in results and logs directories
MODEL_NAME = "gemini-3.1-with-transkribus"

# REVISION: Changed from "gemini-2.0-flash" to "gemini-3.1-pro-preview"
# This is the actual API model identifier sent to the GenAI service
FULL_MODEL_NAME = "gemini-3.1-pro-preview"

# REVISION: Set to 65536 (maximum). Reasoning model thinking tokens count
# against this budget, so we must maximize to avoid truncation.
MAX_OUTPUT_TOKENS = 65536

###############################################################################
# Utility: Time Formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a number of seconds into H:MM:SS for clean logging.

    Args:
        seconds: The elapsed time in seconds.

    Returns:
        A string formatted as HH:MM:SS.
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
    Parse command-line arguments: --pdf and --temperature.

    Arguments:
        --pdf: Name of the PDF file in data/pdfs/ (required)
        --temperature: Sampling temperature for the LLM (default: 0.0)

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Gemini-3.1 post-correction pipeline (using existing Transkribus OCR)."
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
    This is used to determine the next run number for auto-incrementing.

    Args:
        temp_folder: The temperature-specific results directory to scan.

    Returns:
        A list of run numbers (integers) found in the folder.
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
# Utility: Write a JSON log file
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in the logs_revisions/llm_img2txt/<model_name>/ folder.

    The log file is named with a timestamp so each run produces a unique log.

    Args:
        log_dict: Dictionary of run metadata to serialize as JSON.
        model_name: The model name string used for subfolder naming.
    """
    # REVISION: Logs are written to logs_revisions/ instead of logs/
    pipeline_logs_dir = LOGS_DIR / model_name
    pipeline_logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename using the current timestamp
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
    Reads the core Gemini prompt from src/prompts/llm_img2txt/gemini-3.1.txt
    and returns it as a string.

    This prompt contains the base transcription/correction rules that are
    combined with the Transkribus OCR text to form the full prompt.

    Returns:
        The prompt text as a string.
    """
    # REVISION: Changed from "gemini-2.0.txt" to "gemini-3.1.txt"
    prompt_path = PROJECT_ROOT / "src" / "prompts" / "llm_img2txt" / "gemini-3.1.txt"
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

    This function combines:
      1) The base transcription prompt loaded from the prompt file
      2) The pre-existing Transkribus OCR text for the page

    The resulting prompt tells the model to use the OCR text as a reference for
    archaic spellings while correcting any OCR errors based on the page image.

    Args:
        ocr_text: Raw text extracted from OCR (Transkribus output) for this page.
        image_path: Path to the corresponding page image (used for context, not
                    directly included in the prompt text -- the image is sent
                    separately as a content part).

    Returns:
        The full prompt string combining base rules and OCR text.
    """
    # Load the base prompt from the text file
    base_prompt = load_gemini_prompt().strip()

    # Merge the base prompt and the OCR text into a single prompt string
    # The OCR text is appended below the rules so the model can cross-reference
    # the Transkribus output with the actual page image
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
    Main function for the Gemini-3.1 post-correction pipeline.

    Pipeline steps:
      1) Parse arguments and configure logging
      2) Convert PDF to per-page PNG images (if not already present)
      3) Create the run folder structure under results_revisions/
      4) For each page:
         a) Load the pre-existing Transkribus OCR text from results/ocr_img2txt/transkribus/
         b) Build a combined prompt (base rules + OCR text)
         c) Call Gemini 3.1 with the page image + prompt (no retry, exits on failure)
         d) Save the corrected text output
      5) Merge all per-page text files into a single <pdf_stem>.txt
      6) Write a JSON run log with timing and token usage statistics
      7) If any page fails after retries, the pipeline stops immediately
    """
    # -------------------------------------------------------------------------
    # Step 1: Parse arguments and configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf
    temperature = args.temperature

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # REVISION: Updated log header to reference Gemini-3.1
    logging.info("=== Gemini-3.1 Post-Correction Pipeline ===")
    logging.info(f"PDF: {pdf_name}")
    logging.info(f"Temperature: {temperature} (default=0.0)")

    # Check that the API key is available
    if not API_KEY:
        logging.error("GOOGLE_API_KEY not set in .env or environment.")
        sys.exit(1)

    # Start the overall pipeline timer
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 2: Prepare PDF path and create PNG images if needed
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at {pdf_path}")
        sys.exit(1)

    # PNG folder: data/page_by_page/PNG/<pdf_stem>/
    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"Converting PDF -> PNG in {png_dir}")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Use pdf2image to render each page of the PDF as a PNG image
        png_pages = convert_from_path(str(pdf_path))
        for i, page_img in enumerate(png_pages, start=1):
            img_path = png_dir / f"page_{i:04d}.png"
            page_img.save(img_path, "PNG")
        logging.info(f"Created {len(png_pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"PNG folder already exists: {png_dir}")

    # Gather all page PNG files sorted by name
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error("No PNG page images found. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)
    logging.info(f"Total pages to process: {total_pages}")

    # -------------------------------------------------------------------------
    # Step 3: Create results folder structure
    # REVISION: Results go to results_revisions/ instead of results/
    # Path: results_revisions/llm_img2txt/gemini-3.1-with-transkribus/<pdf_stem>/temperature_X/run_XX/page_by_page
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / MODEL_NAME / pdf_stem
    temp_dir = base_results_path / f"temperature_{temperature}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Determine the next run number by scanning existing run directories
    existing_runs = find_existing_runs_in_temperature_folder(temp_dir)
    next_run_number = (max(existing_runs) + 1) if existing_runs else 1

    run_dir = temp_dir / f"run_{str(next_run_number).zfill(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Create the page_by_page subfolder for per-page text output
    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # Step 4: Initialize counters for Gemini token usage tracking
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_tokens = 0

    # -------------------------------------------------------------------------
    # Step 5: Main loop -- for each page, read existing OCR text and call Gemini
    # -------------------------------------------------------------------------
    page_text_files = []

    # The Transkribus OCR text comes from the ORIGINAL results directory (not results_revisions)
    # because the OCR was produced during the original experiment
    transkribus_ocr_root = PROJECT_ROOT / "results" / "ocr_img2txt" / "transkribus" / pdf_stem / "run_01" / "page_by_page"

    for idx, png_path in enumerate(png_files):
        page_num = idx + 1

        logging.info(f"=== Page {page_num} of {total_pages} ===")
        logging.info(f"PNG: {png_path.name}")

        # 5a) Read the existing Transkribus OCR text for this page
        transkribus_txt_path = transkribus_ocr_root / f"page_{page_num:04d}.txt"
        if not transkribus_txt_path.is_file():
            logging.error(f"OCR text file not found: {transkribus_txt_path}")
            sys.exit(1)

        ocr_text = transkribus_txt_path.read_text(encoding='utf-8').strip()
        logging.info(f"Loaded OCR text (~{len(ocr_text)} chars).")

        # 5b) Construct the full prompt: base rules + Transkribus OCR text
        gemini_prompt_text = multimodal_ocr_correction(ocr_text, png_path)

        gemini_output = None
        page_usage_prompt = 0
        page_usage_candidate = 0
        page_usage_total = 0

        logging.info("Calling Gemini for post-correction...")

        # 5c) Call Gemini 3.1 (no retry -- exits on failure)
        tmp_file = None
        try:
            # Save the PNG to a temporary file for uploading to the GenAI service
            with Image.open(png_path) as pil_img, \
                 tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_file = tmp.name
                pil_img.save(tmp_file, "PNG")

            # Upload the image to the GenAI client
            client = genai.Client(api_key=API_KEY)
            file_upload = client.files.upload(file=tmp_file)

            # Send the image + prompt to Gemini 3.1
            # The request content consists of two parts:
            #   1) The page image (uploaded via URI)
            #   2) The combined prompt text (base rules + OCR text)
            # REVISION: Uses FULL_MODEL_NAME "gemini-3.1-pro-preview" instead of "gemini-2.0-flash"
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
                    # REVISION: Dynamic thinking for reasoning model
                    thinking_config=types.ThinkingConfig(thinking_budget=-1),
                ),
            )

            # Validate the response
            if not response or not response.text:
                logging.error(f"Gemini returned empty response for page {page_num}")
                sys.exit(1)

            # Extract the corrected text output
            gemini_output = response.text

            # Record token usage for this page
            usage = response.usage_metadata
            if usage:
                page_usage_prompt = usage.prompt_token_count or 0
                page_usage_candidate = usage.candidates_token_count or 0
                page_usage_total = usage.total_token_count or (
                    page_usage_prompt + page_usage_candidate
                )

            # Accumulate into running totals
            total_prompt_tokens += page_usage_prompt
            total_candidate_tokens += page_usage_candidate
            total_tokens += page_usage_total

        except SystemExit:
            raise
        except Exception as e:
            logging.error(f"Gemini API call failed for page {page_num}: {e}")
            sys.exit(1)
        finally:
            # Clean up the temporary file
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass

        # 5d) Save the Gemini output to a per-page text file
        page_text_path = run_page_dir / f"page_{page_num:04d}.txt"
        with open(page_text_path, 'w', encoding='utf-8') as f:
            f.write(gemini_output)

        page_text_files.append(page_text_path)

        # Log token usage for this page
        logging.info(
            f"Gemini usage for page {page_num}: prompt={page_usage_prompt}, "
            f"candidate={page_usage_candidate}, total={page_usage_total}"
        )
        logging.info(
            f"Accumulated usage so far: prompt={total_prompt_tokens}, "
            f"candidate={total_candidate_tokens}, total={total_tokens}"
        )

        # Log timing and estimated time remaining
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
    # Step 6: Merge all per-page text files into a single <pdf_stem>.txt
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path}")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            # Read each page's corrected text, strip whitespace, separate with blank lines
            text_content = txt_file.read_text(encoding='utf-8').strip()
            outf.write(text_content + "\n\n")

    # -------------------------------------------------------------------------
    # Step 7: Write JSON log with run metadata
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start_time
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature": temperature,
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
    # Step 8: Final usage summary
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
