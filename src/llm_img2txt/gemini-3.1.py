#!/usr/bin/env python3
"""
###############################################################################
# REVISION EXPERIMENT SCRIPT
#
# This script is an adaptation of the Gemini 2.0 img2txt pipeline for the
# Gemini 3.1 Pro Preview model. It is part of a revision experiment that
# stores results and logs in separate directories (results_revisions/ and
# logs_revisions/) to keep them isolated from the original experiment outputs.
#
# Changes from gemini-2.0.py:
#   - MODEL_NAME changed from "gemini-2.0" to "gemini-3.1"
#   - FULL_MODEL_NAME changed from "gemini-2.0-flash" to "gemini-3.1-pro-preview"
#   - RESULTS_DIR changed from "results/" to "results_revisions/"
#   - LOGS_DIR changed from "logs/" to "logs_revisions/"
#   - Prompt file changed from "gemini-2.0.txt" to "gemini-3.1.txt"
#
# All pipeline logic remains identical to the original.
###############################################################################

Gemini-3.1 PDF -> PNG -> TEXT Pipeline

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls Gemini-3.1 for each page image, producing text output.
     - If an API call fails or returns empty, the script logs the error and exits immediately.
  3) Merges all returned page texts into a single TXT file (<pdf_stem>.txt).
  4) Logs usage tokens (prompt/candidate) per page, accumulates them across all pages, and saves a JSON run log.

Everything is aligned with the other Gemini pipeline scripts for consistency.
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
from typing import Optional, List

# Google-GenAI (Gemini) library -- used for both Gemini 2.0 and 3.1 models
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# Project Paths
###############################################################################
# PROJECT_ROOT is two levels up from this script (src/llm_img2txt/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where source PDFs and page images are stored
DATA_DIR = PROJECT_ROOT / "data"

# Directory containing the prompt text files for the img2txt pipeline
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_img2txt"

# REVISION: Changed from "results" to "results_revisions" to isolate revision outputs
RESULTS_DIR = PROJECT_ROOT / "results_revisions" / "llm_img2txt"

# REVISION: Changed from "logs" to "logs_revisions" to isolate revision logs
LOGS_DIR = PROJECT_ROOT / "logs_revisions" / "llm_img2txt"

# Path to the .env file containing API keys
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load environment variables
###############################################################################
# Load the .env file so we can access GOOGLE_API_KEY
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")  # Must match your .env key

###############################################################################
# Constants
###############################################################################
# REVISION: Changed from "gemini-2.0" to "gemini-3.1" -- used for folder naming and prompt file lookup
MODEL_NAME = "gemini-3.1"

# REVISION: Changed from "gemini-2.0-flash" to "gemini-3.1-pro-preview" -- the actual API model identifier
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
    Parse command-line arguments for the Gemini-3.1 PDF-to-text pipeline.

    Arguments:
        --pdf: Name of the PDF file in data/pdfs/ (required)
        --temperature: Sampling temperature for the LLM (default: 0.0)

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Gemini-3.1 PDF-to-text pipeline")

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
        help="Temperature for the LLM call (default: 0.0)"
    )

    return parser.parse_args()

###############################################################################
# Utility: Find existing run_XY directories to auto-increment run number
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
# Main Pipeline
###############################################################################
def main() -> None:
    """
    Main function for Gemini-3.1 PDF-to-text pipeline.

    Pipeline steps:
      1) Parse arguments and configure logging
      2) Load the transcription prompt from src/prompts/llm_img2txt/gemini-3.1.txt
      3) Convert the PDF to per-page PNG images in data/page_by_page/PNG/<pdf_stem>
      4) Create the run folder structure under results_revisions/llm_img2txt/gemini-3.1/...
      5) For each page: call Gemini 3.1 with the image + prompt, save per-page text
      6) Concatenate all per-page text files into a single <pdf_stem>.txt
      7) Write a JSON run log with timing and token usage statistics
    """
    # -------------------------------------------------------------------------
    # Step 1: Parse arguments and configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf
    temperature = args.temperature

    # Configure logging to write to stdout with timestamps
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # REVISION: Updated log header to reference Gemini-3.1
    logging.info("=== Gemini-3.1 PDF -> PNG -> TEXT Pipeline ===")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model: {MODEL_NAME}, Full model: {FULL_MODEL_NAME}")
    logging.info(f"Temperature: {temperature}")

    # Start the overall pipeline timer
    overall_start = time.time()

    # -------------------------------------------------------------------------
    # Step 2: Load the transcription prompt
    # REVISION: Prompt file is now "gemini-3.1.txt" instead of "gemini-2.0.txt"
    # -------------------------------------------------------------------------
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)

    # Read the prompt text that will be sent with each page image
    transcription_prompt = prompt_path.read_text(encoding='utf-8').strip()
    if not transcription_prompt:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)
    logging.info(f"Prompt loaded from: {prompt_path}")

    # -------------------------------------------------------------------------
    # Step 3: Convert PDF to per-page PNG images (skip if already done)
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    # PNG images are stored in data/page_by_page/PNG/<pdf_stem>/
    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"No PNG folder found; converting PDF -> PNG in {png_dir} ...")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Use pdf2image to render each page as a PNG image
        pages = convert_from_path(str(pdf_path))
        for i, page_img in enumerate(pages, start=1):
            img_path = png_dir / f"page_{i:04d}.png"
            page_img.save(img_path, "PNG")
        logging.info(f"Created {len(pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"Folder {png_dir} already exists; skipping PDF->PNG step.")

    # Gather all PNG files sorted by name
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error(f"No page images found in {png_dir}. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)

    # -------------------------------------------------------------------------
    # Step 4: Create results folder structure
    # REVISION: Results go to results_revisions/ instead of results/
    # Path: results_revisions/llm_img2txt/gemini-3.1/<pdf_stem>/temperature_x.x/run_nn/page_by_page
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
    # Step 5: Process each page -- call Gemini 3.1 (no retry)
    # -------------------------------------------------------------------------
    # Initialize token usage accumulators
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0

    # Track successfully created text files for final concatenation
    page_text_files = []

    for idx, png_path in enumerate(png_files, start=1):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        try:
            with Image.open(png_path) as pil_image:
                # Log image dimensions and DPI for debugging/auditing
                width, height = pil_image.size
                dpi_value = pil_image.info.get("dpi", None)
                if dpi_value and len(dpi_value) == 2:
                    logging.info(
                        f"Image metadata -> width={width}px, height={height}px, dpi={dpi_value}"
                    )
                else:
                    logging.info(
                        f"Image metadata -> width={width}px, height={height}px, dpi=UNKNOWN"
                    )

                # Single API call attempt (no retry -- exits on failure)
                transcription = None
                page_usage_prompt = 0
                page_usage_candidate = 0
                page_usage_total = 0

                tmp_file = None
                try:
                    # Save the PIL image to a temporary file for upload to the GenAI service
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_file = tmp.name
                        pil_image.save(tmp_file, "PNG")

                    # Create a GenAI client and upload the image
                    client = genai.Client(api_key=API_KEY)
                    file_upload = client.files.upload(file=tmp_file)

                    # Send the image + prompt to Gemini 3.1
                    # REVISION: Uses FULL_MODEL_NAME "gemini-3.1-pro-preview" instead of "gemini-2.0-flash"
                    response = client.models.generate_content(
                        model=FULL_MODEL_NAME,
                        contents=[
                            # The uploaded page image
                            types.Part.from_uri(
                                file_uri=file_upload.uri,
                                mime_type=file_upload.mime_type,
                            ),
                            # The transcription prompt text
                            transcription_prompt
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
                        logging.error(
                            f"Gemini-3.1 returned empty response for page {idx}"
                        )
                        sys.exit(1)

                    # Extract the transcribed text
                    transcription = response.text
                    usage = response.usage_metadata

                    # Record token usage for this page
                    page_usage_prompt = usage.prompt_token_count or 0
                    page_usage_candidate = usage.candidates_token_count or 0
                    page_usage_total = usage.total_token_count or (
                        page_usage_prompt + page_usage_candidate
                    )

                    # Accumulate into running totals
                    total_prompt_tokens += page_usage_prompt
                    total_candidates_tokens += page_usage_candidate
                    total_tokens += page_usage_total

                except SystemExit:
                    raise
                except Exception as e:
                    logging.error(f"Gemini-3.1 API call failed for page {idx}: {e}")
                    sys.exit(1)
                finally:
                    # Clean up the temporary file
                    if tmp_file and os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except:
                            pass

        except SystemExit:
            raise
        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            sys.exit(1)

        # Log token usage for this page
        logging.info(
            f"Gemini-3.1 usage for page {idx}: "
            f"input={page_usage_prompt}, candidate={page_usage_candidate}, total={page_usage_total}"
        )
        logging.info(
            f"Accumulated so far: input={total_prompt_tokens}, "
            f"candidate={total_candidates_tokens}, total={total_tokens}"
        )

        # Save the transcribed text for this page
        page_text_path = run_page_dir / f"{png_path.stem}.txt"
        with open(page_text_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        page_text_files.append(page_text_path)

        # Log timing and estimated time remaining
        elapsed = time.time() - overall_start
        pages_done = idx
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

    logging.info("All pages processed. Individual text files created.")

    # -------------------------------------------------------------------------
    # Step 6: Concatenate all per-page text files into a single output file
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")

    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for text_file in sorted(page_text_files):
            # Read each page's text, strip whitespace, and separate pages with blank lines
            outf.write(Path(text_file).read_text(encoding='utf-8').strip())
            outf.write("\n\n")  # separate pages with a blank line

    logging.info(f"Final concatenated file: {final_txt_path}")

    # -------------------------------------------------------------------------
    # Step 7: Write a JSON log summarizing the run
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature": temperature,
        "run_directory": str(run_dir),
        "prompt_file": str(prompt_path),
        "pages_count": len(page_text_files),
        "final_text_file": str(final_txt_path),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidate_tokens": total_candidates_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }
    write_json_log(log_info, MODEL_NAME)

    # Final usage summary
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Total input (prompt) tokens: {total_prompt_tokens}")
    logging.info(f"Total candidate tokens: {total_candidates_tokens}")
    logging.info(f"Grand total tokens: {total_tokens}")

    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()
