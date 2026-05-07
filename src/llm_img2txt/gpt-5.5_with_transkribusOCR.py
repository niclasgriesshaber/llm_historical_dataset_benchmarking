#!/usr/bin/env python3
"""
GPT-5.5 Post-Correction Pipeline (reading existing Transkribus OCR text).

# REVISION: This script is adapted from gpt-4o_with_transkribusOCR.py for the
# GPT-5.5 model. All structural logic is identical; only model identifiers,
# result/log directory paths, and prompt file references have been updated.
# The Transkribus OCR source path remains unchanged (reads from results/,
# NOT results_revisions/) because the OCR output is a fixed input.

This script:
  1) Ensures each page of the PDF has a PNG image in data/page_by_page/PNG/<pdf_stem>.
     - If not found, it converts the PDF into per-page PNG images.
  2) For each page:
     - Loads existing OCR text from:
       results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
     - Calls GPT-5.5 (single attempt) passing:
       - The page image (PNG).
       - The existing OCR text, as context in a "post-correction" prompt.
     - Stores the GPT-5.5 output as page_000N.txt in a new run folder.
  3) Merges all per-page GPT-5.5 outputs into a single TXT file (<pdf_stem>.txt).
  4) Logs usage tokens (prompt/candidate) per page, accumulates them, and saves a JSON run log.
  5) If GPT-5.5 fails on any page, the script stops immediately.

Example usage:
  ./gpt-5.5_with_transkribusOCR.py --pdf type-1.pdf [--temperature 0.0]
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
from typing import List, Optional

import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# Project Paths
# These paths define where input data, results, and logs are stored.
# PROJECT_ROOT is computed relative to this script's location (two levels up).
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
# REVISION: Changed from "results" to "results_revisions" for GPT-5.5 output isolation
RESULTS_DIR = PROJECT_ROOT / "results_revisions" / "llm_img2txt"
# REVISION: Changed from "logs" to "logs_revisions" for GPT-5.5 log isolation
LOGS_DIR = PROJECT_ROOT / "logs_revisions" / "llm_img2txt"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# We store PNG in data/page_by_page/PNG/<pdf_stem>
# Transkribus OCR text is pre-existing in:
# results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
# NOTE: The Transkribus OCR source stays in the OLD results/ directory (not results_revisions/)
# because it is a fixed, pre-computed input that does not change between model revisions.

###############################################################################
# Load environment variables (for GPT-5.5)
# Reads the .env file to obtain the OpenAI API key needed for authentication.
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("OPENAI_API_KEY")  # Must match your .env key

###############################################################################
# GPT-5.5 Model Config
# REVISION: Changed from gpt-4o to gpt-5.5 for both MODEL_NAME and FULL_MODEL_NAME
###############################################################################
# REVISION: Changed from "gpt-4o-with-transkribus" to "gpt-5.5-with-transkribus"
MODEL_NAME = "gpt-5.5-with-transkribus"     # Folder name for final outputs
# REVISION: Changed from "gpt-4o-2024-05-13" to "gpt-5.5"
FULL_MODEL_NAME = "gpt-5.5"                 # Full GPT-5.5 model ID
# REVISION: Set to 65536 (maximum). Reasoning tokens count against this budget.
MAX_OUTPUT_TOKENS = 65536
SEED = 42

###############################################################################
# Utility: Time Formatting
# Converts raw seconds into a human-readable H:MM:SS string for log output.
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
# Defines the CLI interface: --pdf (required) and --temperature (optional).
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments: --pdf and --temperature
    """
    parser = argparse.ArgumentParser(
        # REVISION: Changed description from "GPT-4o" to "GPT-5.5"
        description="GPT-5.5 post-correction pipeline (using existing Transkribus OCR)."
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
        # REVISION: Changed help text from "GPT-4o" to "GPT-5.5"
        help="Temperature for the GPT-5.5 LLM call (default: 0.0)"
    )
    return parser.parse_args()

###############################################################################
# Utility: Find existing 'run_XX' directories to auto-increment
# Scans a temperature-specific folder for run_XX subdirectories and returns
# the list of run numbers as integers, used to auto-increment run numbering.
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
# Utility: Write a JSON log file
# Saves a timestamped JSON log file containing run metadata, usage stats,
# and timing information to the logs directory for this model.
# REVISION: Logs are written to logs_revisions/ instead of logs/
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in the logs_revisions/llm_img2txt/<model_name>/ folder.
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
# Load the GPT-5.5 prompt from file
# Reads the strict transcription prompt that will be combined with the OCR text
# to form the full post-correction prompt sent to the model.
# REVISION: Changed prompt filename from "gpt-4o.txt" to "gpt-5.5.txt"
###############################################################################
def load_gpt55_prompt() -> str:
    """
    Reads the strict transcription prompt from:
    src/prompts/llm_img2txt/gpt-5.5.txt
    and returns it as a string.
    """
    # REVISION: Changed from "gpt-4o.txt" to "gpt-5.5.txt"
    prompt_path = PROJECT_ROOT / "src" / "prompts" / "llm_img2txt" / "gpt-5.5.txt"
    if not prompt_path.is_file():
        logging.error(f"Prompt file not found at {prompt_path}")
        sys.exit(1)
    return prompt_path.read_text(encoding='utf-8')

###############################################################################
# Multimodal OCR correction prompt function
# Combines the base transcription prompt with the pre-existing Transkribus OCR
# text to create a full post-correction prompt. The OCR text provides spelling
# context for archaic words, while the image allows visual verification.
###############################################################################
def multimodal_ocr_correction(ocr_text: str, image_path: Path) -> str:
    """
    Applies multimodal OCR post-correction while enforcing strict transcription rules.

    :param ocr_text: Raw text extracted from OCR (Transkribus output)
    :param image_path: Path to the corresponding image for verification (not used directly in prompt text)
    :return: Full prompt to pass to GPT-5.5
    """
    # Load the base prompt from the text file
    # REVISION: Changed from load_gpt4o_prompt() to load_gpt55_prompt()
    base_prompt = load_gpt55_prompt().strip()

    # Merge the base prompt and the OCR text into a single prompt string
    # The OCR output is appended as context to help the model correct errors
    gpt55_prompt_text = (
        f"{base_prompt}\n\n"
        "Below is the OCR output from Transkribus so you know how to spell the archaic words. Please use this information to correct any errors and ensure the text is fully compliant with the strict transcription rules.\n"
        "-- OCR Output (Transkribus) --\n"
        f"{ocr_text}\n"
    )

    return gpt55_prompt_text

###############################################################################
# OpenAI GPT-5.5 API call
# Sends a prompt and a base64-encoded PNG image to the OpenAI chat completions
# endpoint. Returns the generated text and token usage statistics.
###############################################################################
def openai_api(
    prompt: str,
    pil_image: Image.Image,
    full_model_name: str,
    max_tokens: int,
    temperature: float,
    api_key: str
) -> (Optional[str], dict):
    """
    Call OpenAI's GPT-5.5 with an image + text prompt, returning (text_out, usage_dict).

    usage_dict has e.g. {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}.
    If there's an error, return (None, {}).
    """

    import base64
    from io import BytesIO

    # Convert image to RGB mode if needed, then encode as base64 PNG
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    with BytesIO() as buffer:
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

    # Set up HTTP headers with JSON content type and Bearer token authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Build the API request payload with the model name, message content
    # (text prompt + base64 image), and generation parameters
    payload = {
        "model": full_model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        # REVISION: GPT-5.5 is a reasoning model — uses max_completion_tokens,
        # does not support temperature or seed parameters.
        "max_completion_tokens": max_tokens,
        # REVISION: Dynamic reasoning (comparable to Gemini's thinking_budget=-1)
        "reasoning_effort": "high",
    }

    # Send the request to the OpenAI chat completions endpoint
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            # Successful response: extract the generated text and usage statistics
            data = response.json()
            text_out = data["choices"][0]["message"]["content"]
            usage_info = data.get("usage", {})
            return text_out, usage_info
        else:
            # Non-200 status: raise an error with the status code and response body
            # REVISION: Changed error message from "GPT-4o" to "GPT-5.5"
            raise ValueError(f"OpenAI GPT-5.5 error {response.status_code}: {response.text}")
    except Exception as e:
        # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
        logging.error(f"OpenAI GPT-5.5 call failed: {e}")
        return None, {}

###############################################################################
# Main Pipeline
###############################################################################
def main() -> None:
    """
    1) Convert PDF => PNG (if not already present) in data/page_by_page/PNG/<pdf_stem>.
    2) For each page:
       - Load pre-existing OCR text from:
         results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
       - Call GPT-5.5 => post-correct using the OCR text + image (single attempt).
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

    # REVISION: Changed pipeline banner from "GPT-4o" to "GPT-5.5"
    logging.info("=== GPT-5.5 Post-Correction Pipeline ===")
    logging.info(f"PDF: {pdf_name}")
    logging.info(f"Temperature: {temperature} (default=0.0) | Seed={SEED}")

    # Check environment for GPT-5.5 API key
    if not API_KEY:
        logging.error("OPENAI_API_KEY not set in .env or environment.")
        sys.exit(1)

    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # 2. Prepare PDF path & create PNG if needed
    # If the PNG directory already exists, skip the conversion to save time.
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at {pdf_path}")
        sys.exit(1)

    # PNG folder for page images
    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"Converting PDF -> PNG in {png_dir}")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Use pdf2image to convert each PDF page to a PIL Image, then save as PNG
        png_pages = convert_from_path(str(pdf_path))
        for i, page_img in enumerate(png_pages, start=1):
            img_path = png_dir / f"page_{i:04d}.png"
            page_img.save(img_path, "PNG")
        logging.info(f"Created {len(png_pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"PNG folder already exists: {png_dir}")

    # Gather all PNG page files sorted by filename
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error("No PNG page images found. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)
    logging.info(f"Total pages to process: {total_pages}")

    # -------------------------------------------------------------------------
    # 3. Create results folder
    # Directory structure:
    #   results_revisions/llm_img2txt/gpt-5.5-with-transkribus/<pdf_stem>/temperature_X/run_XX/page_by_page
    # Each run gets its own numbered subdirectory for reproducibility.
    # REVISION: Changed from results/ to results_revisions/
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / MODEL_NAME / pdf_stem
    temp_dir = base_results_path / f"temperature_{temperature}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Determine the next run number by scanning for existing run_XX directories
    existing_runs = find_existing_runs_in_temperature_folder(temp_dir)
    next_run_number = (max(existing_runs) + 1) if existing_runs else 1

    run_dir = temp_dir / f"run_{str(next_run_number).zfill(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Create the page_by_page subdirectory for individual page text files
    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # 4. Initialize counters for GPT-5.5 usage
    # Track prompt, candidate, and total tokens across all pages.
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_tokens = 0

    # -------------------------------------------------------------------------
    # 5. Main Loop: For each page, read existing OCR text -> call GPT-5.5
    # For each page image, load the corresponding Transkribus OCR text,
    # build a post-correction prompt, call GPT-5.5, and save the output.
    # -------------------------------------------------------------------------
    page_text_files = []

    # Root folder for pre-existing Transkribus OCR text
    # NOTE: This path intentionally uses "results/" (not "results_revisions/")
    # because the Transkribus OCR output is a fixed, pre-computed input
    # that does not change between model revisions.
    transkribus_ocr_root = (
        PROJECT_ROOT
        / "results"
        / "ocr_img2txt"
        / "transkribus"
        / pdf_stem
        / "run_01"
        / "page_by_page"
    )

    for idx, png_path in enumerate(png_files, start=1):
        page_num = idx
        logging.info(f"=== Page {page_num} of {total_pages} ===")
        logging.info(f"PNG: {png_path.name}")

        # 5a) Read the existing OCR text from the Transkribus results directory
        transkribus_txt_path = transkribus_ocr_root / f"page_{page_num:04d}.txt"
        if not transkribus_txt_path.is_file():
            logging.error(f"OCR text file not found: {transkribus_txt_path}")
            sys.exit(1)

        ocr_text = transkribus_txt_path.read_text(encoding='utf-8').strip()
        logging.info(f"Loaded OCR text (~{len(ocr_text)} chars).")

        # 5b) Construct the post-correction prompt by combining the base prompt
        # from gpt-5.5.txt with the Transkribus OCR text as additional context
        post_correction_prompt = multimodal_ocr_correction(ocr_text, png_path)

        gpt55_output = None
        page_usage_prompt = 0
        page_usage_candidate = 0
        page_usage_total = 0

        # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
        logging.info("Calling GPT-5.5 for post-correction...")

        # Single GPT-5.5 call (no retry)
        try:
            with Image.open(png_path) as pil_img:
                returned_text, usage_dict = openai_api(
                    prompt=post_correction_prompt,
                    pil_image=pil_img,
                    full_model_name=FULL_MODEL_NAME,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=temperature,
                    api_key=API_KEY
                )
        except Exception as e:
            logging.error(f"GPT-5.5 call failed for page {page_num}: {e}")
            sys.exit(1)

        if not returned_text:
            logging.error(f"API call failed for page {page_num}")
            sys.exit(1)

        # Gather usage info
        gpt55_output = returned_text
        page_usage_prompt = usage_dict.get("prompt_tokens", 0)
        page_usage_candidate = usage_dict.get("completion_tokens", 0)
        page_usage_total = usage_dict.get(
            "total_tokens",
            page_usage_prompt + page_usage_candidate
        )

        # Update global usage accumulators
        total_prompt_tokens += page_usage_prompt
        total_candidate_tokens += page_usage_candidate
        total_tokens += page_usage_total

        # 5c) Save the GPT-5.5 output to a per-page text file
        page_text_path = run_page_dir / f"page_{page_num:04d}.txt"
        with open(page_text_path, 'w', encoding='utf-8') as f:
            f.write(gpt55_output)

        page_text_files.append(page_text_path)

        # Log usage statistics for this page and cumulative totals
        # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
        logging.info(
            f"GPT-5.5 usage for page {page_num}: prompt={page_usage_prompt}, "
            f"candidate={page_usage_candidate}, total={page_usage_total}"
        )
        logging.info(
            f"Accumulated usage so far: prompt={total_prompt_tokens}, "
            f"candidate={total_candidate_tokens}, total={total_tokens}"
        )

        # Calculate and log timing estimates (elapsed, estimated total, estimated remaining)
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

    # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
    logging.info("All pages processed successfully. GPT-5.5 outputs saved.")

    # -------------------------------------------------------------------------
    # 6. Merge all page text files -> final <pdf_stem>.txt
    # Combines all per-page text outputs into a single file, separated by
    # double newlines, for easy downstream consumption.
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path}")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            text_content = txt_file.read_text(encoding='utf-8').strip()
            outf.write(text_content + "\n\n")

    # -------------------------------------------------------------------------
    # 7. Write JSON Log
    # Captures all run parameters, paths, usage stats, and timing info.
    # REVISION: Logs are written to logs_revisions/ instead of logs/
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
    # Print final usage statistics and total duration to the console.
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
