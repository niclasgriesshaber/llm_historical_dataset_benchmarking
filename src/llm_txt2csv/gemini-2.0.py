#!/usr/bin/env python3
"""
Gemini-2.0 TXT -> JSON -> CSV Pipeline

This script:
  1) Reads a text file from data/ground_truth/txt/<txt_file>.
  2) Concatenates that text below the standard prompt in src/prompts/llm_txt2csv/gemini-2.0.txt.
  3) Calls Gemini-2.0 via the Google GenAI client, retrieving JSON output.
     - Retries up to 1 hour on any error (including JSON parse errors).
  4) Converts the returned JSON to a single CSV in results/llm_txt2csv/gemini-2.0/<txt_stem>/temperature_<T>/run_<NN>/<txt_stem>.csv.
  5) Logs usage tokens and timing, storing a JSON run log in logs/llm_txt2csv/gemini-2.0/run_<timestamp>.json.
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

# Google-GenAI (Gemini-2.0) library
import google.genai as genai
from google.genai import types

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "ground_truth" / "txt"
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "llm_txt2csv" / "gemini-2.0.txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_txt2csv" / "gemini-2.0"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_txt2csv" / "gemini-2.0"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# Model constants
MODEL_NAME = "gemini-2.0"        # For folder naming
FULL_MODEL_NAME = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 8192
RETRY_LIMIT_SECONDS = 3600  # up to 1 hour max for a successful call

###############################################################################
# Utility: Time formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a number of seconds into H:MM:SS for cleaner logging.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

###############################################################################
# Utility: Parse JSON from Gemini-2.0 text response
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON if present; otherwise, fall back to raw text.
    Then parse it as JSON. Raises ValueError if parsing fails.
    """
    fenced_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        response_text,
        re.IGNORECASE,
    )
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        # fallback to entire response, removing any stray backticks
        candidate = response_text.strip().strip("`")

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
    """
    import csv

    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        # fallback: treat as single row with "value" column
        records = [{"value": str(json_data)}]

    # Gather all keys
    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    # ID first (if present), rest sorted
    fieldnames = []
    if "id" in all_keys:
        fieldnames.append("id")

    other_keys = [k for k in sorted(all_keys) if k != "id"]
    fieldnames.extend(other_keys)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            if not isinstance(rec, dict):
                # fallback: place entire object in "value"
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                writer.writerow(row_data)
                continue

            row_data = {}
            for fn in fieldnames:
                row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)

###############################################################################
# Gemini-2.0 API Call with up to 1-hour total retry
###############################################################################
def gemini_api_call(prompt: str, temperature: float) -> Optional[dict]:
    """
    Call Gemini-2.0 with the given prompt (text only), retry up to 1 hour.
    Returns a dict:
      {
        "text": <the text response>,
        "usage": <usage metadata object>
      }
    or None if it fails after 1 hour.
    """
    client = genai.Client(api_key=API_KEY)
    start_retry = time.time()

    while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
        try:
            response = client.models.generate_content(
                model=FULL_MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                ),
            )

            if not response:
                logging.warning("Gemini-2.0 returned an empty response; retrying...")
                continue

            text_candidate = response.text
            if not text_candidate:
                logging.warning("Gemini-2.0 returned no text in the response; retrying...")
                continue

            usage = response.usage_metadata
            return {
                "text": text_candidate,
                "usage": usage
            }

        except Exception as e:
            logging.warning(f"Gemini-2.0 call failed: {e}. Retrying...")

    logging.error("Gemini-2.0 call did not succeed after 1 hour.")
    return None

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the Gemini-2.0 TXT -> JSON -> CSV pipeline.
    """
    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Gemini-2.0 TXT-to-JSON-to-CSV Pipeline")
    parser.add_argument(
        "--txt",
        required=True,
        help="Name of the TXT file in data/ground_truth/txt/, e.g. type-1.txt"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for Gemini-2.0 (default = 0.0)"
    )
    args = parser.parse_args()
    txt_name = args.txt
    temperature = args.temperature

    # -------------------------------------------------------------------------
    # Configure logging
    # -------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== Gemini-2.0 TXT -> JSON -> CSV Pipeline ===")
    logging.info(f"TXT: {txt_name} | Temperature: {temperature}")

    # -------------------------------------------------------------------------
    # Verify the TXT file exists
    # -------------------------------------------------------------------------
    txt_path = DATA_DIR / txt_name
    if not txt_path.is_file():
        logging.error(f"TXT file not found at: {txt_path}")
        sys.exit(1)

    txt_stem = txt_path.stem

    # -------------------------------------------------------------------------
    # Load the task prompt for Gemini-2.0
    # -------------------------------------------------------------------------
    if not PROMPT_PATH.is_file():
        logging.error(f"Missing prompt file: {PROMPT_PATH}")
        sys.exit(1)

    base_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not base_prompt:
        logging.error(f"Prompt file is empty: {PROMPT_PATH}")
        sys.exit(1)

    logging.info(f"Loaded Gemini-2.0 prompt from: {PROMPT_PATH}")

    # -------------------------------------------------------------------------
    # Read the input text file, concatenate below prompt
    # -------------------------------------------------------------------------
    user_text = txt_path.read_text(encoding="utf-8").strip()
    full_prompt = f"{base_prompt}\n\n{user_text}"

    # -------------------------------------------------------------------------
    # Prepare results folder => results/llm_txt2csv/gemini-2.0/<txt_stem>/temperature_x.x/run_nn/
    # -------------------------------------------------------------------------
    txt_folder = RESULTS_DIR / txt_stem
    temp_folder = txt_folder / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Collect existing runs
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
    # Start timing
    # -------------------------------------------------------------------------
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Make a single Gemini-2.0 call with up to 1-hour retry
    # -------------------------------------------------------------------------
    result = gemini_api_call(prompt=full_prompt, temperature=temperature)
    if result is None:
        # If no success after 1 hour, we abort
        total_duration = time.time() - overall_start_time
        logging.error(f"Call failed after 1 hour. Elapsed: {format_duration(total_duration)}")
        sys.exit(1)

    response_text = result["text"]
    usage_meta = result["usage"]

    page_prompt_tokens = usage_meta.prompt_token_count or 0
    page_candidate_tokens = usage_meta.candidates_token_count or 0
    page_total_tokens = usage_meta.total_token_count or 0

    logging.info(
        f"Gemini-2.0 usage: input={page_prompt_tokens}, "
        f"candidate={page_candidate_tokens}, total={page_total_tokens}"
    )

    # -------------------------------------------------------------------------
    # Retry JSON parsing up to 1 hour if needed
    # -------------------------------------------------------------------------
    parse_start_time = time.time()
    while True:
        try:
            parsed = parse_json_str(response_text)
        except ValueError as ve:
            if (time.time() - parse_start_time) > RETRY_LIMIT_SECONDS:
                logging.error("Skipping due to JSON parse failure after 1 hour.")
                parsed = None
                break
            logging.error(f"JSON parse error: {ve}")
            logging.error("Retrying Gemini-2.0 call to fix parse issues...")
            new_result = gemini_api_call(prompt=full_prompt, temperature=temperature)
            if not new_result:
                logging.error("Could not fix JSON parse after an additional attempt. Exiting.")
                parsed = None
                break

            response_text = new_result["text"]
            usage_retry = new_result["usage"]

            # Accumulate usage
            page_prompt_tokens += (usage_retry.prompt_token_count or 0)
            page_candidate_tokens += (usage_retry.candidates_token_count or 0)
            page_total_tokens += (usage_retry.total_token_count or 0)

            logging.info(
                f"[Retry usage] Additional tokens: "
                f"input={usage_retry.prompt_token_count}, "
                f"candidate={usage_retry.candidates_token_count}, "
                f"total={usage_retry.total_token_count}"
            )
            logging.info(
                f"New accumulated: input={page_prompt_tokens}, "
                f"candidate={page_candidate_tokens}, total={page_total_tokens}"
            )
        else:
            break  # parse succeeded

    if not parsed:
        # If JSON parse never succeeded
        total_duration = time.time() - overall_start_time
        logging.error(f"Final parse failure. Elapsed time: {format_duration(total_duration)}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Save the raw JSON
    # -------------------------------------------------------------------------
    raw_json_path = run_folder / f"{txt_stem}.json"
    with raw_json_path.open("w", encoding="utf-8") as jf:
        json.dump(parsed, jf, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # If the parsed data is a list or dict, consider labeling each item with an 'id'
    # -------------------------------------------------------------------------
    def add_ids_to_data(data: Any) -> Any:
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
            # single scalar
            return {"id": 1, "value": str(data)}

    final_data = add_ids_to_data(parsed)

    # -------------------------------------------------------------------------
    # Convert JSON to CSV
    # -------------------------------------------------------------------------
    final_csv_path = run_folder / f"{txt_stem}.csv"
    convert_json_to_csv(final_data, final_csv_path)
    logging.info(f"Final CSV saved at: {final_csv_path}")

    # -------------------------------------------------------------------------
    # Log run metadata
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
    # Final summary
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Input tokens used: {page_prompt_tokens}")
    logging.info(f"Candidate tokens used: {page_candidate_tokens}")
    logging.info(f"Grand total tokens used: {page_total_tokens}")
    logging.info(f"Pipeline completed in {format_duration(total_duration)}.")
    logging.info("All done!")


if __name__ == "__main__":
    main()