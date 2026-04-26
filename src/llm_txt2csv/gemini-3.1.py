#!/usr/bin/env python3
"""
###############################################################################
# REVISION EXPERIMENT SCRIPT
#
# This script is an adaptation of the Gemini 2.0 txt2csv pipeline for the
# Gemini 3.1 Pro Preview model. It is part of a revision experiment that
# stores results and logs in separate directories (results_revisions/ and
# logs_revisions/) to keep them isolated from the original experiment outputs.
#
# Changes from gemini-2.0.py:
#   - MODEL_NAME changed from "gemini-2.0" to "gemini-3.1"
#   - FULL_MODEL_NAME changed from "gemini-2.0-flash" to "gemini-3.1-pro-preview"
#   - PROMPT_PATH changed to reference "gemini-3.1.txt" instead of "gemini-2.0.txt"
#   - RESULTS_DIR changed from "results/" to "results_revisions/"
#   - LOGS_DIR changed from "logs/" to "logs_revisions/"
#
# All pipeline logic remains identical to the original.
###############################################################################

Gemini-3.1 TXT -> JSON -> CSV Pipeline

This script:
  1) Reads a text file from data/ground_truth/txt/<txt_file>.
  2) Concatenates that text below the standard prompt in src/prompts/llm_txt2csv/gemini-3.1.txt.
  3) Calls Gemini-3.1 via the Google GenAI client, retrieving JSON output.
     - If any API call or JSON parse fails, the script logs the error and exits immediately.
  4) Converts the returned JSON to a single CSV in results_revisions/llm_txt2csv/gemini-3.1/<txt_stem>/temperature_<T>/run_<NN>/<txt_stem>.csv.
  5) Logs usage tokens and timing, storing a JSON run log in logs_revisions/llm_txt2csv/gemini-3.1/run_<timestamp>.json.
"""

import os
import sys
import re
import json
import time
import argparse
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Dict, List, Optional

from dotenv import load_dotenv

# Google-GenAI (Gemini) library -- used for both Gemini 2.0 and 3.1 models
import google.genai as genai
from google.genai import types

###############################################################################
# Project Paths
###############################################################################
# PROJECT_ROOT is two levels up from this script (src/llm_txt2csv/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory containing the ground truth text files that serve as input
DATA_DIR = PROJECT_ROOT / "data" / "ground_truth" / "txt"

# REVISION: Changed from "gemini-2.0.txt" to "gemini-3.1.txt" to use the revision prompt
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "llm_txt2csv" / "gemini-3.1.txt"

# REVISION: Changed from "results" to "results_revisions" to isolate revision outputs
# Also includes "gemini-3.1" subfolder instead of "gemini-2.0"
RESULTS_DIR = PROJECT_ROOT / "results_revisions" / "llm_txt2csv" / "gemini-3.1"

# REVISION: Changed from "logs" to "logs_revisions" to isolate revision logs
# Also includes "gemini-3.1" subfolder instead of "gemini-2.0"
LOGS_DIR = PROJECT_ROOT / "logs_revisions" / "llm_txt2csv" / "gemini-3.1"

# Path to the .env file containing API keys
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
###############################################################################
# Load the .env file so we can access GOOGLE_API_KEY
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# REVISION: Changed from "gemini-2.0" to "gemini-3.1" -- used for folder naming
MODEL_NAME = "gemini-3.1"

# REVISION: Changed from "gemini-2.0-flash" to "gemini-3.1-pro-preview" -- the actual API model identifier
FULL_MODEL_NAME = "gemini-3.1-pro-preview"

# REVISION: Set to 65536 (maximum). Reasoning model thinking tokens count
# against this budget, so we must maximize to avoid truncation.
MAX_OUTPUT_TOKENS = 65536

###############################################################################
# Utility: Time formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a number of seconds into H:MM:SS for cleaner logging.

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
# Utility: Parse JSON from Gemini text response
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON if present; otherwise, fall back to raw text.
    Then parse it as JSON. Raises ValueError if parsing fails.

    The model sometimes wraps its JSON output in markdown code fences like:
        ```json
        { ... }
        ```
    This function handles both fenced and unfenced responses.

    Args:
        response_text: The raw text response from the Gemini API.

    Returns:
        The parsed JSON object (dict, list, or primitive).

    Raises:
        ValueError: If the text cannot be parsed as valid JSON.
    """
    # Try to find JSON inside markdown code fences first
    fenced_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        response_text,
        re.IGNORECASE,
    )
    if fenced_match:
        # Extract the content between the code fences
        candidate = fenced_match.group(1).strip()
    else:
        # No code fences found -- treat the entire response as potential JSON,
        # stripping any stray backticks that the model might have included
        candidate = response_text.strip().strip("`")

    # Parse and return the JSON string
    return json.loads(candidate)

###############################################################################
# Utility: Convert JSON to CSV
###############################################################################
def convert_json_to_csv(json_data: Union[Dict, List], csv_path: Path) -> None:
    """
    Flatten JSON objects/arrays into a CSV at csv_path.
    1) If top-level is a dict, that's 1 row.
    2) If top-level is a list, each element is a row.
    3) Reorder columns so 'id' (if present) is near the front, and keep any
       other fields in alphabetical order.

    Args:
        json_data: The parsed JSON data (dict or list of dicts).
        csv_path: The file path where the CSV should be written.
    """
    import csv

    # Normalize to a list of records
    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        # Fallback: treat as single row with "value" column
        records = [{"value": str(json_data)}]

    # Collect all unique keys across all records to form CSV column headers
    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    # Place "id" first if present, then sort remaining keys alphabetically
    fieldnames = []
    if "id" in all_keys:
        fieldnames.append("id")

    other_keys = [k for k in sorted(all_keys) if k != "id"]
    fieldnames.extend(other_keys)

    # Ensure the output directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the CSV file
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            if not isinstance(rec, dict):
                # Fallback: place entire object in "value" column
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                writer.writerow(row_data)
                continue

            # Build a row using the fieldnames, defaulting missing keys to empty string
            row_data = {}
            for fn in fieldnames:
                row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)

###############################################################################
# Gemini 3.1 API Call (no retry)
###############################################################################
def gemini_api_call(prompt: str, temperature: float) -> dict:
    """
    Call Gemini 3.1 with the given prompt (text only). If the call fails or
    returns empty results, log the error and exit immediately.

    This pipeline processes text (not images), so no file upload is needed.
    The prompt text (base instructions + ground truth text) is sent directly.

    Args:
        prompt: The full prompt text to send to the model.
        temperature: The sampling temperature for the model.

    Returns:
        A dict with "text" (the model's response) and "usage" (token usage metadata).
    """
    # Create a GenAI client using the API key from environment
    client = genai.Client(api_key=API_KEY)

    try:
        # Send the text-only prompt to Gemini 3.1
        # REVISION: Uses FULL_MODEL_NAME "gemini-3.1-pro-preview" instead of "gemini-2.0-flash"
        response = client.models.generate_content(
            model=FULL_MODEL_NAME,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                # Request JSON output format from the model
                response_mime_type="application/json",
                # REVISION: Dynamic thinking for reasoning model
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            ),
        )

        # Validate the response
        if not response:
            logging.error("Gemini-3.1 returned an empty response")
            sys.exit(1)

        text_candidate = response.text
        if not text_candidate:
            logging.error("Gemini-3.1 returned no text in the response")
            sys.exit(1)

        # Extract token usage metadata for logging
        usage = response.usage_metadata
        return {
            "text": text_candidate,
            "usage": usage
        }

    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"Gemini-3.1 API call failed: {e}")
        sys.exit(1)

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the Gemini-3.1 TXT -> JSON -> CSV pipeline.

    Pipeline steps:
      1) Parse command-line arguments (--txt, --temperature)
      2) Configure logging to stdout
      3) Verify the specified TXT file exists in data/ground_truth/txt/
      4) Load the task prompt from src/prompts/llm_txt2csv/gemini-3.1.txt
      5) Concatenate the prompt with the input text
      6) Create a run folder under results_revisions/llm_txt2csv/gemini-3.1/...
      7) Call Gemini 3.1 API (text-only, no image), parse JSON response
      8) Save raw JSON and convert to CSV
      9) Write a JSON run log to logs_revisions/llm_txt2csv/gemini-3.1/
    """
    # -------------------------------------------------------------------------
    # Step 1: Parse command-line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Gemini-3.1 TXT-to-JSON-to-CSV Pipeline")
    parser.add_argument(
        "--txt",
        required=True,
        help="Name of the TXT file in data/ground_truth/txt/, e.g. type-1.txt"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for Gemini-3.1 (default = 0.0)"
    )
    args = parser.parse_args()
    txt_name = args.txt
    temperature = args.temperature

    # -------------------------------------------------------------------------
    # Step 2: Configure logging to stdout with timestamps
    # -------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # REVISION: Updated log header to reference Gemini-3.1
    logging.info("=== Gemini-3.1 TXT -> JSON -> CSV Pipeline ===")
    logging.info(f"TXT: {txt_name} | Temperature: {temperature}")

    # -------------------------------------------------------------------------
    # Step 3: Verify the input TXT file exists
    # -------------------------------------------------------------------------
    txt_path = DATA_DIR / txt_name
    if not txt_path.is_file():
        logging.error(f"TXT file not found at: {txt_path}")
        sys.exit(1)

    # Extract the stem (filename without extension) for naming output files/folders
    txt_stem = txt_path.stem

    # -------------------------------------------------------------------------
    # Step 4: Load the task prompt for Gemini-3.1
    # REVISION: Prompt file is now "gemini-3.1.txt" instead of "gemini-2.0.txt"
    # -------------------------------------------------------------------------
    if not PROMPT_PATH.is_file():
        logging.error(f"Missing prompt file: {PROMPT_PATH}")
        sys.exit(1)

    # Read the base prompt that contains the extraction instructions
    base_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not base_prompt:
        logging.error(f"Prompt file is empty: {PROMPT_PATH}")
        sys.exit(1)

    logging.info(f"Loaded Gemini-3.1 prompt from: {PROMPT_PATH}")

    # -------------------------------------------------------------------------
    # Step 5: Read the input text file and concatenate it below the prompt
    # The full prompt = base instructions + "\n\n" + ground truth text
    # -------------------------------------------------------------------------
    user_text = txt_path.read_text(encoding="utf-8").strip()
    full_prompt = f"{base_prompt}\n\n{user_text}"

    # -------------------------------------------------------------------------
    # Step 6: Prepare results folder structure
    # REVISION: Results go to results_revisions/ instead of results/
    # Path: results_revisions/llm_txt2csv/gemini-3.1/<txt_stem>/temperature_x.x/run_nn/
    # -------------------------------------------------------------------------
    txt_folder = RESULTS_DIR / txt_stem
    temp_folder = txt_folder / f"temperature_{temperature}"
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
    next_run = highest_run_num + 1
    run_folder = temp_folder / f"run_{str(next_run).zfill(2)}"
    run_folder.mkdir(parents=True, exist_ok=False)
    logging.info(f"Created run folder: {run_folder}")

    # -------------------------------------------------------------------------
    # Start timing the API call
    # -------------------------------------------------------------------------
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 7: Make a single Gemini 3.1 API call (no retry)
    # This is a text-only call (no image upload needed)
    # -------------------------------------------------------------------------
    result = gemini_api_call(prompt=full_prompt, temperature=temperature)

    # Extract the text response and token usage metadata
    response_text = result["text"]
    usage_meta = result["usage"]

    # Record token usage
    page_prompt_tokens = usage_meta.prompt_token_count or 0
    page_candidate_tokens = usage_meta.candidates_token_count or 0
    page_total_tokens = usage_meta.total_token_count or 0

    logging.info(
        f"Gemini-3.1 usage: input={page_prompt_tokens}, "
        f"candidate={page_candidate_tokens}, total={page_total_tokens}"
    )

    # -------------------------------------------------------------------------
    # Parse the JSON from the model's text response (no retry)
    # If parsing fails, log the error and exit immediately
    # -------------------------------------------------------------------------
    try:
        parsed = parse_json_str(response_text)
    except ValueError as ve:
        logging.error(f"JSON parse error: {ve}")
        logging.error(f"Response text: \n{response_text}\n")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Save the raw JSON output to a file in the run folder
    # -------------------------------------------------------------------------
    raw_json_path = run_folder / f"{txt_stem}.json"
    with raw_json_path.open("w", encoding="utf-8") as jf:
        json.dump(parsed, jf, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Add sequential "id" fields to each record if not already present
    # This ensures each row in the CSV has a unique identifier
    # -------------------------------------------------------------------------
    def add_ids_to_data(data: Any) -> Any:
        """
        Add 'id' field to parsed JSON data if not already present.

        - Single dict: assign id=1
        - List of dicts: assign id=1,2,3,... to each item
        - Scalar: wrap in a dict with id=1 and value=str(data)

        Args:
            data: The parsed JSON data.

        Returns:
            The data with 'id' fields added.
        """
        if isinstance(data, dict):
            if "id" not in data:
                data["id"] = 1
            return data
        elif isinstance(data, list):
            updated = []
            for i, item in enumerate(data, start=1):
                if isinstance(item, dict):
                    if "id" not in item:
                        item["id"] = i
                    updated.append(item)
                else:
                    updated.append({"id": i, "value": str(item)})
            return updated
        else:
            # Single scalar value
            return {"id": 1, "value": str(data)}

    final_data = add_ids_to_data(parsed)

    # -------------------------------------------------------------------------
    # Step 8: Convert the JSON data to CSV format
    # -------------------------------------------------------------------------
    final_csv_path = run_folder / f"{txt_stem}.csv"
    convert_json_to_csv(final_data, final_csv_path)
    logging.info(f"Final CSV saved at: {final_csv_path}")

    # -------------------------------------------------------------------------
    # Step 9: Write JSON log with run metadata
    # REVISION: Logs go to logs_revisions/ instead of logs/
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start_time
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp_str}.json"
    log_path = LOGS_DIR / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "txt_name": txt_name,
        "txt_path": str(txt_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature": temperature,
        "run_directory": str(run_folder),
        "prompt_file": str(PROMPT_PATH),
        "final_csv": str(final_csv_path),
        "json_output": str(raw_json_path),
        "usage": {
            "prompt_tokens": page_prompt_tokens,
            "candidate_tokens": page_candidate_tokens,
            "total_tokens": page_total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }

    with log_path.open("w", encoding="utf-8") as lf:
        json.dump(log_data, lf, indent=4)

    # -------------------------------------------------------------------------
    # Final usage summary
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Input tokens used: {page_prompt_tokens}")
    logging.info(f"Candidate tokens used: {page_candidate_tokens}")
    logging.info(f"Grand total tokens used: {page_total_tokens}")
    logging.info(f"Pipeline completed in {format_duration(total_duration)}.")
    logging.info("All done!")


if __name__ == "__main__":
    main()
