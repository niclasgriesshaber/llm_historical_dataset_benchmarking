#!/usr/bin/env python3
"""
GPT-5.5 PDF -> PNG -> TEXT Pipeline

# REVISION: This script is adapted from gpt-4o.py for the GPT-5.5 model.
# All structural logic is identical; only model identifiers, result/log
# directory paths, and prompt file references have been updated.

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls GPT-5.5 for each page image, retrieving text output.
     - If the API call fails or returns empty, the script logs the error and exits.
  3) Merges all returned page texts into a single TXT file (<pdf_stem>.txt).
  4) Logs usage tokens per page, accumulates them across all pages, and saves a JSON run log.
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

import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# Project Paths
# These paths define where input data, prompts, results, and logs are stored.
# PROJECT_ROOT is computed relative to this script's location (two levels up).
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_img2txt"
# REVISION: Changed from "results" to "results_revisions" for GPT-5.5 output isolation
RESULTS_DIR = PROJECT_ROOT / "results_revisions" / "llm_img2txt"
# REVISION: Changed from "logs" to "logs_revisions" for GPT-5.5 log isolation
LOGS_DIR = PROJECT_ROOT / "logs_revisions" / "llm_img2txt"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load environment variables
# Reads the .env file to obtain the OpenAI API key needed for authentication.
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("OPENAI_API_KEY")  # Must match your .env key

###############################################################################
# Model Constants
# REVISION: Changed from gpt-4o to gpt-5.5 for both MODEL_NAME and FULL_MODEL_NAME
###############################################################################
# REVISION: Changed from "gpt-4o" to "gpt-5.5"
MODEL_NAME = "gpt-5.5"
# REVISION: Changed from "gpt-4o-2024-05-13" to "gpt-5.5"
FULL_MODEL_NAME = "gpt-5.5"
# REVISION: Set to 65536 (maximum). Reasoning tokens count against this budget.
MAX_OUTPUT_TOKENS = 65536
SEED = 42  # Not used by OpenAI, but included for consistency

###############################################################################
# Utility: Time Formatting
# Converts raw seconds into a human-readable H:MM:SS string for log output.
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a number of seconds into H:MM:SS for consistent logging.
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
    Parse command-line arguments for a GPT-5.5 PDF-to-text pipeline.
    """
    # REVISION: Changed description from "GPT-4o" to "GPT-5.5"
    parser = argparse.ArgumentParser(description="GPT-5.5 PDF-to-text pipeline")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. example.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the LLM call (default 0.0)"
    )
    return parser.parse_args()

###############################################################################
# Utility: Find existing run_XY directories
# Scans a temperature-specific folder for run_XX subdirectories and returns
# the list of run numbers as integers, used to auto-increment run numbering.
###############################################################################
def find_existing_runs_in_temperature_folder(temp_folder: Path) -> List[int]:
    """
    Scan for run_XX subfolders in temp_folder, returning a list of integers.
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
# Utility: Write a JSON log
# Saves a timestamped JSON log file containing run metadata, usage stats,
# and timing information to the logs directory for this model.
# REVISION: Logs are written to logs_revisions/ instead of logs/
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in logs_revisions/llm_img2txt/<model_name>/.
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
# OpenAI GPT-5.5 Call
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
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
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
        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, json=payload)
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
# Main
###############################################################################
def main() -> None:
    """
    Main GPT-5.5 PDF -> PNG -> TEXT pipeline with:
      - data/page_by_page/PNG/<pdf_stem> for images
      - single API call per page (no retry)
      - usage token logging (input, candidate, total)
      - final text concatenation
      - JSON log & final usage summary
    """
    # -------------------------------------------------------------------------
    # Parse arguments, configure logging
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
    logging.info("=== GPT-5.5 PDF -> PNG -> TEXT Pipeline ===")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model: {MODEL_NAME}, Full model: {FULL_MODEL_NAME}")
    logging.info(f"Temperature: {temperature} | Seed={SEED}")

    # Overall timing start
    overall_start = time.time()

    # -------------------------------------------------------------------------
    # Load transcription prompt
    # The prompt file contains instructions for how the model should transcribe
    # the page images into plain text.
    # REVISION: Changed prompt filename from "gpt-4o.txt" to "gpt-5.5.txt"
    # -------------------------------------------------------------------------
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)

    transcription_prompt = prompt_path.read_text(encoding='utf-8').strip()
    if not transcription_prompt:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)

    logging.info(f"Prompt loaded from: {prompt_path}")

    # -------------------------------------------------------------------------
    # Convert PDF -> PNG in data/page_by_page/PNG/<pdf_stem>
    # If the PNG directory already exists, skip the conversion to save time.
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"No PNG folder found; converting PDF -> PNG in {png_dir} ...")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Use pdf2image to convert each PDF page to a PIL Image, then save as PNG
        pages = convert_from_path(str(pdf_path))
        for i, page_img in enumerate(pages, start=1):
            out_png = png_dir / f"page_{i:04d}.png"
            page_img.save(out_png, "PNG")
        logging.info(f"Created {len(pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"Folder {png_dir} already exists; skipping PDF->PNG step.")

    # Gather all PNG files sorted by filename (page_0001.png, page_0002.png, ...)
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error(f"No PNG pages found in {png_dir}. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)

    # -------------------------------------------------------------------------
    # Prepare results folder
    # Directory structure: results_revisions/llm_img2txt/gpt-5.5/<pdf_stem>/temperature_x.x/run_nn/page_by_page
    # Each run gets its own numbered subdirectory for reproducibility.
    # REVISION: Changed from results/ to results_revisions/ and gpt-4o to gpt-5.5
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / MODEL_NAME / pdf_stem
    temp_folder = base_results_path / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Determine the next run number by scanning for existing run_XX directories
    existing_runs = find_existing_runs_in_temperature_folder(temp_folder)
    next_run = max(existing_runs) + 1 if existing_runs else 1
    run_dir = temp_folder / f"run_{str(next_run).zfill(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Create the page_by_page subdirectory for individual page text files
    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # Accumulate usage
    # Track prompt, completion, and total tokens across all pages for the run log.
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_completion_tokens = 0  # "candidate" tokens in some other scripts
    total_tokens = 0

    page_text_files = []

    # -------------------------------------------------------------------------
    # Process each PNG
    # For each page image, call GPT-5.5 (single attempt), save the text output,
    # accumulate token usage, and log progress with time estimates.
    # -------------------------------------------------------------------------
    for idx, png_path in enumerate(png_files, start=1):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        # Open the image and log its metadata (dimensions and DPI)
        try:
            with Image.open(png_path) as pil_image:
                width, height = pil_image.size
                dpi_value = pil_image.info.get("dpi", None)
                if dpi_value and len(dpi_value) == 2:
                    logging.info(f"Image metadata -> width={width}px, height={height}px, dpi={dpi_value}")
                else:
                    logging.info(f"Image metadata -> width={width}px, height={height}px, dpi=UNKNOWN")

                # Single GPT-5.5 call (no retry)
                text_out = None
                page_prompt = 0
                page_candidate = 0
                page_total = 0

                returned_text, usage_dict = openai_api(
                    prompt=transcription_prompt,
                    pil_image=pil_image,
                    full_model_name=FULL_MODEL_NAME,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=temperature,
                    api_key=API_KEY
                )

                if not returned_text:
                    logging.error(f"API call failed for page {idx}")
                    sys.exit(1)

                # Successful response: capture text and usage stats
                text_out = returned_text
                page_prompt = usage_dict.get("prompt_tokens", 0)
                page_candidate = usage_dict.get("completion_tokens", 0)
                page_total = usage_dict.get("total_tokens", page_prompt + page_candidate)

                # Update cumulative accumulators
                total_prompt_tokens += page_prompt
                total_completion_tokens += page_candidate
                total_tokens += page_total

        except SystemExit:
            raise
        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            sys.exit(1)

        # Log usage statistics for this page and cumulative totals
        # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
        logging.info(
            f"GPT-5.5 usage for page {idx}: "
            f"input={page_prompt}, candidate={page_candidate}, total={page_total}"
        )
        logging.info(
            f"Accumulated so far: input={total_prompt_tokens}, "
            f"candidate={total_completion_tokens}, total={total_tokens}"
        )

        # Save the page-level text output to a .txt file in the page_by_page directory
        page_txt_path = run_page_dir / f"{png_path.stem}.txt"
        page_txt_path.write_text(text_out, encoding='utf-8')
        page_text_files.append(page_txt_path)

        # Calculate and log timing estimates (elapsed, estimated total, estimated remaining)
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
    # Concatenate page texts => <pdf_stem>.txt in run_dir
    # Combines all per-page text outputs into a single file, separated by
    # double newlines, for easy downstream consumption.
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            content = txt_file.read_text(encoding='utf-8').strip()
            outf.write(content + "\n\n")

    logging.info(f"Final concatenated file: {final_txt_path}")

    # -------------------------------------------------------------------------
    # Write JSON log
    # Captures all run parameters, paths, usage stats, and timing info.
    # REVISION: Logs are written to logs_revisions/ instead of logs/
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature": temperature,
        "seed": SEED,
        "run_directory": str(run_dir),
        "prompt_file": str(prompt_path),
        "pages_count": len(page_text_files),
        "final_text_file": str(final_txt_path),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidate_tokens": total_completion_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }
    write_json_log(log_info, MODEL_NAME)

    # Final usage summary printed to console
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Total input (prompt) tokens used: {total_prompt_tokens}")
    logging.info(f"Total candidate tokens used: {total_completion_tokens}")
    logging.info(f"Grand total of all tokens used: {total_tokens}")

    # Final log line with total duration
    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()
