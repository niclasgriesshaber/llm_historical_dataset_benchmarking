#!/usr/bin/env python3
"""
GPT-5.5 PDF -> PNG -> JSON -> CSV Pipeline

# REVISION: This script is adapted from gpt-4o.py for the GPT-5.5 model.
# All structural logic is identical; only model identifiers, result/log
# directory paths, and prompt file references have been updated.

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls GPT-5.5 for each page image, retrieving JSON output.
     - If the API call fails or returns invalid JSON, the script logs the error and exits.
  3) Merges all returned JSON data into a single CSV (named <pdf_stem>.csv).
  4) Logs usage tokens per page (prompt/completion), accumulates them across all pages, and saves a JSON run log.
"""

import os
import sys
import re
import json
import time
import argparse
import logging
import tempfile
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Dict, List, Optional

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# OpenAI GPT-5.5 call function
# This function sends a prompt and an image to the OpenAI chat completions
# endpoint and returns the model's text response along with token usage info.
# It encodes the PIL image as a base64 PNG and embeds it in the API payload.
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
    Call OpenAI's GPT-5.5 with an image + text prompt.
    Returns (text_out, usage_info) or (None, {}).

    usage_info is a dict with keys: {prompt_tokens, completion_tokens, total_tokens}.
    """
    import base64
    from io import BytesIO

    # Ensure the image is in RGB mode so it can be saved as PNG without issues
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert the PIL image to a base64-encoded PNG string for the API payload
    with BytesIO() as buffer:
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

    # Set up HTTP headers with JSON content type and Bearer token authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
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
        # REVISION: GPT-5.5 is a reasoning model — uses max_completion_tokens
        # instead of max_tokens, and does not support temperature or seed parameters.
        "max_completion_tokens": max_tokens,
        # REVISION: Dynamic reasoning — "low" lets the model decide how much
        # thinking to use (comparable to Gemini's thinking_budget=-1).
        "reasoning_effort": "high",
    }

    # Send the request to the OpenAI chat completions endpoint and parse the response
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=payload
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
# Project Paths
# These paths define where input data, prompts, results, and logs are stored.
# PROJECT_ROOT is computed relative to this script's location (two levels up).
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_img2csv"
# REVISION: Changed from "results" to "results_revisions" for GPT-5.5 output isolation
RESULTS_DIR = PROJECT_ROOT / "results_revisions" / "llm_img2csv"
# REVISION: Changed from "logs" to "logs_revisions" for GPT-5.5 log isolation
LOGS_DIR = PROJECT_ROOT / "logs_revisions" / "llm_img2csv"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
# Reads the .env file to obtain the OpenAI API key needed for authentication.
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model constants
# REVISION: Changed from gpt-4o to gpt-5.5
MODEL_NAME = "gpt-5.5"                    # Short name for folder naming
# REVISION: Changed from "gpt-4o-2024-08-06" to "gpt-5.5"
FULL_MODEL_NAME = "gpt-5.5"               # Full GPT-5.5 model ID
# REVISION: Set to 65536 (maximum). GPT-5.5 is a reasoning model — internal
# reasoning tokens count against this budget, so we maximize to avoid truncation.
MAX_OUTPUT_TOKENS = 65536
SEED = 42                                 # For consistency (not enforced by OpenAI)

###############################################################################
# Utility: Time formatting
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
# Utility: Parse JSON from GPT-5.5 text response
# Extracts JSON from a code-fenced block (```json ... ```) if present,
# otherwise tries to parse the raw text directly as JSON.
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON if present, otherwise fallback to raw text.
    Then parse it as JSON. Raises ValueError if parsing fails.
    """
    # Look for a fenced code block containing JSON
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        # No fence found; strip any stray backticks and attempt direct parse
        candidate = response_text.strip().strip("`")
    return json.loads(candidate)

###############################################################################
# Utility: Reorder dictionary keys with page_number at the end
# Ensures that 'page_number' and 'additional_information' appear as the last
# keys in each record dict for consistent CSV column ordering.
###############################################################################
def reorder_dict_with_page_number(d: Dict[str, Any], page_number: int) -> Dict[str, Any]:
    """
    Return a new dict that includes a 'page_number' key at the end
    and ensures 'additional_information' is last if present.
    """
    special_keys = {"page_number", "additional_information"}
    # Collect all non-special keys and sort them alphabetically
    base_keys = [k for k in d.keys() if k not in special_keys]
    base_keys.sort()

    out = {}
    # Add all regular keys first in sorted order
    for k in base_keys:
        out[k] = d[k]

    # Append page_number as the next-to-last key
    out["page_number"] = page_number

    # Append additional_information last if it was in the original dict
    if "additional_information" in d:
        out["additional_information"] = d["additional_information"]

    return out

###############################################################################
# Utility: Convert JSON to CSV
# Flattens a list of JSON objects (or a single object) into a CSV file.
# Columns are ordered alphabetically, with 'page_number' and
# 'additional_information' placed at the end.
###############################################################################
def convert_json_to_csv(json_data: Union[Dict, List], csv_path: Path) -> None:
    """
    Flatten JSON objects/arrays into a CSV at csv_path.
    1) If top-level is a single dict, that's 1 row.
    2) If top-level is a list, each element is a row.
    3) Reorder columns so 'page_number' & 'additional_information' come last.
    """
    import csv

    # Normalize input: wrap a single dict in a list for uniform handling
    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        records = [{"value": str(json_data)}]

    # Collect the union of all keys across all records
    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    # Build fieldnames: alphabetical base keys, then special keys at the end
    special_order = ["page_number", "additional_information"]
    base_keys = [k for k in all_keys if k not in special_order]
    base_keys.sort()
    fieldnames = base_keys + [k for k in special_order if k in all_keys]

    # Write the CSV file, creating parent directories if needed
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            if not isinstance(rec, dict):
                # Non-dict records are written as a single "value" column
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                writer.writerow(row_data)
                continue

            # Build a row dict, using empty string for any missing keys
            row_data = {}
            for fn in fieldnames:
                row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)

###############################################################################
# GPT-5.5 API Call (single attempt, no retry)
###############################################################################
def gpt4o_api_call(
    prompt: str,
    pil_image: Image.Image,
    temperature: float
) -> Optional[dict]:
    """
    Call GPT-5.5 with the given prompt + image (single attempt, no retry).
    Returns a dict:
      {
        "text": <the text response>,
        "usage": <usage metadata with {prompt_tokens, completion_tokens, total_tokens}>
      }
    or None if the call fails or returns empty text.
    """
    text_out, usage_info = openai_api(
        prompt=prompt,
        pil_image=pil_image,
        full_model_name=FULL_MODEL_NAME,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        api_key=OPENAI_API_KEY
    )

    if not text_out:
        return None

    return {
        "text": text_out,
        "usage": usage_info
    }

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the GPT-5.5 PDF-to-CSV pipeline.
    Orchestrates the full workflow: argument parsing, PDF-to-PNG conversion,
    per-page GPT-5.5 API calls, JSON parsing, CSV generation, and logging.
    """
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    # --pdf: the PDF filename located in data/pdfs/
    # --temperature: LLM sampling temperature (default 0.0 for deterministic output)
    # --continue_from_page: optionally resume from a specific page in the latest run
    # -------------------------------------------------------------------------
    # REVISION: Changed description from "GPT-4o" to "GPT-5.5"
    parser = argparse.ArgumentParser(description="GPT-5.5 PDF-to-JSON-to-CSV Pipeline")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. my_file.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        # REVISION: Changed help text from "GPT-4o" to "GPT-5.5"
        help="LLM temperature for GPT-5.5 (default = 0.0)"
    )
    parser.add_argument(
        "--continue_from_page",
        type=int,
        default=None,
        help=(
            "If provided, continue from this page number in the highest existing run folder. "
            "If not provided, a new run folder is always created."
        )
    )
    args = parser.parse_args()
    pdf_name = args.pdf
    temperature = args.temperature
    continue_from_page = args.continue_from_page

    # -------------------------------------------------------------------------
    # Configure logging
    # Sets up INFO-level logging to stdout with timestamp and level prefix.
    # -------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # REVISION: Changed pipeline banner from "GPT-4o" to "GPT-5.5"
    logging.info("=== GPT-5.5 PDF -> PNG -> JSON -> CSV Pipeline ===")
    logging.info(f"PDF: {pdf_name} | Temperature: {temperature} | Seed={SEED}")
    logging.info(f"Continue from page: {continue_from_page if continue_from_page else 'None (create new run)'}")

    # -------------------------------------------------------------------------
    # Verify the PDF exists on disk before proceeding
    # -------------------------------------------------------------------------
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    pdf_stem = Path(pdf_name).stem

    # -------------------------------------------------------------------------
    # Load the task prompt for GPT-5.5
    # The prompt file contains instructions for the model on how to extract
    # structured JSON data from the page images.
    # -------------------------------------------------------------------------
    # REVISION: Changed prompt filename from "gpt-4o.txt" to "gpt-5.5.txt"
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Missing prompt file: {prompt_path}")
        sys.exit(1)

    task_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not task_prompt:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)

    # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
    logging.info(f"Loaded GPT-5.5 prompt from: {prompt_path}")

    # -------------------------------------------------------------------------
    # Check / create data/page_by_page/PNG/<pdf_stem> for PNG generation
    # If the PNG directory does not yet exist, convert all PDF pages to PNGs.
    # If it already exists, skip the conversion step to save time.
    # -------------------------------------------------------------------------
    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"No PNG folder found; converting PDF -> PNG in {png_dir} ...")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Use pdf2image to convert each PDF page to a PIL Image, then save as PNG
        pages = convert_from_path(str(pdf_path))
        for i, pil_img in enumerate(pages, start=1):
            out_png = png_dir / f"page_{i:04d}.png"
            pil_img.save(out_png, "PNG")
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
    # Directory structure: results_revisions/llm_img2csv/gpt-5.5/<pdf_name>/temperature_x.x/run_nn/
    # Each run gets its own numbered subdirectory for reproducibility.
    # REVISION: Changed from results/ to results_revisions/ and gpt-4o to gpt-5.5
    # -------------------------------------------------------------------------
    pdf_folder = RESULTS_DIR / MODEL_NAME / pdf_name.split(".")[0]
    temp_folder = pdf_folder / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Scan for existing run_XX directories to determine the next run number
    existing_runs = []
    for child in temp_folder.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                run_num = int(child.name.split("_")[1])
                existing_runs.append(run_num)
            except ValueError:
                pass
    highest_run_num = max(existing_runs) if existing_runs else 0

    # If --continue_from_page is not provided => always create a new run folder
    if continue_from_page is None:
        next_run = highest_run_num + 1
        run_folder = temp_folder / f"run_{str(next_run).zfill(2)}"
        run_folder.mkdir(parents=True, exist_ok=False)
        logging.info(f"Creating new run folder: {run_folder}")
    else:
        # If user explicitly set --continue_from_page, reuse highest existing run folder
        if highest_run_num == 0:
            run_folder = temp_folder / "run_01"
            run_folder.mkdir(parents=True, exist_ok=True)
            logging.info(
                f"No existing runs found, but --continue_from_page={continue_from_page} set. Using {run_folder}."
            )
        else:
            run_folder = temp_folder / f"run_{str(highest_run_num).zfill(2)}"
            logging.info(
                f"Continuing from page {continue_from_page}, using existing run folder: {run_folder}"
            )

    # Subfolder for per-page JSON output files
    run_page_json_dir = run_folder / "page_by_page"
    run_page_json_dir.mkdir(parents=True, exist_ok=True)

    # Final CSV path -> run_folder/<pdf_stem>.csv
    final_csv_path = run_folder / f"{pdf_stem}.csv"

    # -------------------------------------------------------------------------
    # Accumulators for usage tokens
    # Track prompt, completion, and total tokens across all pages for the run log.
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # -------------------------------------------------------------------------
    # Start timing the entire pipeline
    # -------------------------------------------------------------------------
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Process each page image from `continue_from_page` onward if specified
    # If --continue_from_page was given, skip pages before that index.
    # -------------------------------------------------------------------------
    start_index = 1
    if continue_from_page is not None and continue_from_page > 1:
        start_index = continue_from_page
    # Adjust the png_files list to start from the requested page index
    png_files = png_files[start_index - 1:]

    for idx, png_path in enumerate(png_files, start=start_index):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        # Log image metadata (dimensions and DPI) before calling GPT-5.5
        try:
            with Image.open(png_path) as pil_image:
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

                # Call GPT-5.5 (single attempt, no retry)
                result = gpt4o_api_call(
                    prompt=task_prompt,
                    pil_image=pil_image,
                    temperature=temperature
                )
        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            sys.exit(1)

        if result is None:
            logging.error(f"API call failed for page {idx}")
            sys.exit(1)

        response_text = result["text"]
        usage_meta = result["usage"]

        # ---------------------------------------------------------------------
        # Update usage accumulators for this page
        # These track cumulative token consumption for the entire run.
        # ---------------------------------------------------------------------
        page_prompt_tokens = usage_meta.get("prompt_tokens", 0)
        page_completion_tokens = usage_meta.get("completion_tokens", 0)
        page_total_tokens = usage_meta.get("total_tokens", page_prompt_tokens + page_completion_tokens)

        total_prompt_tokens += page_prompt_tokens
        total_completion_tokens += page_completion_tokens
        total_tokens += page_total_tokens

        # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
        logging.info(
            f"GPT-5.5 usage for page {idx}: "
            f"input={page_prompt_tokens}, completion={page_completion_tokens}, total={page_total_tokens}"
        )
        logging.info(
            f"Accumulated usage so far: input={total_prompt_tokens}, "
            f"completion={total_completion_tokens}, total={total_tokens}"
        )

        # ---------------------------------------------------------------------
        # JSON parsing (single attempt, no retry)
        # ---------------------------------------------------------------------
        try:
            parsed = parse_json_str(response_text)
        except ValueError as ve:
            logging.error(f"JSON parse error for page {idx}: {ve}")
            sys.exit(1)

        # ---------------------------------------------------------------------
        # Save page-level JSON
        # Each page's parsed JSON is written to its own file for traceability.
        # ---------------------------------------------------------------------
        page_json_path = run_page_json_dir / f"{png_path.stem}.json"
        with page_json_path.open("w", encoding="utf-8") as jf:
            json.dump(parsed, jf, indent=2, ensure_ascii=False)

        # ---------------------------------------------------------------------
        # Timing / Estimation
        # Calculate elapsed time, average per page, and estimated remaining time.
        # ---------------------------------------------------------------------
        elapsed = time.time() - overall_start_time
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

    # -------------------------------------------------------------------------
    # Gather all JSON files from run_page_json_dir
    # After all pages are processed, merge per-page JSON into a single dataset.
    # -------------------------------------------------------------------------
    logging.info("Gathering all JSON files from page_by_page folder to build final CSV...")
    all_page_json_files = sorted(run_page_json_dir.glob("page_*.json"))

    merged_data: List[Any] = []

    for fpath in all_page_json_files:
        # Parse page number from the filename, e.g. 'page_0001.json' => '0001' => 1
        page_str = fpath.stem.split("_")[1]
        page_num = int(page_str)

        with fpath.open("r", encoding="utf-8") as jf:
            content = json.load(jf)

        # Reorder dictionary keys so that 'page_number' is at the end
        if isinstance(content, list):
            for obj in content:
                if isinstance(obj, dict):
                    merged_data.append(reorder_dict_with_page_number(obj, page_num))
                else:
                    merged_data.append(obj)
        elif isinstance(content, dict):
            merged_data.append(reorder_dict_with_page_number(content, page_num))
        else:
            merged_data.append(content)

    # -------------------------------------------------------------------------
    # Add "id" to each record
    # Assigns a sequential integer ID to every record for downstream use.
    # -------------------------------------------------------------------------
    for i, record in enumerate(merged_data, start=1):
        if isinstance(record, dict):
            record["id"] = i
        else:
            merged_data[i-1] = {"id": i, "value": str(record)}

    # -------------------------------------------------------------------------
    # Convert merged_data to CSV
    # Write the final combined dataset as a CSV file in the run directory.
    # -------------------------------------------------------------------------
    convert_json_to_csv(merged_data, final_csv_path)
    logging.info(f"Final CSV (all pages) saved at: {final_csv_path}")

    # -------------------------------------------------------------------------
    # Write JSON log with run metadata
    # Captures all run parameters, paths, usage stats, and timing info.
    # REVISION: Logs are written to logs_revisions/ instead of logs/
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start_time
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp_str}.json"
    log_path = LOGS_DIR / MODEL_NAME / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature": temperature,
        "run_directory": str(run_folder),
        "prompt_file": str(prompt_path),
        "pages_count": total_pages,
        "page_json_directory": str(run_page_json_dir),
        "final_csv": str(final_csv_path),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
        "seed": SEED
    }

    with log_path.open("w", encoding="utf-8") as lf:
        json.dump(log_data, lf, indent=4)

    # -------------------------------------------------------------------------
    # Final usage summary
    # Print cumulative token usage to the console for quick reference.
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Total input (prompt) tokens used: {total_prompt_tokens}")
    logging.info(f"Total completion tokens used: {total_completion_tokens}")
    logging.info(f"Grand total of all tokens used: {total_tokens}")

    # -------------------------------------------------------------------------
    # Final logging
    # Report total pipeline duration and signal completion.
    # -------------------------------------------------------------------------
    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()
