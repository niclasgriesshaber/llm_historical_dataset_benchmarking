#!/usr/bin/env python3
"""
GPT-5.5 TXT -> JSON -> CSV Pipeline

# REVISION: This script is adapted from gpt-4o.py for the GPT-5.5 model.
# All structural logic is identical; only model identifiers, result/log
# directory paths, and prompt file references have been updated.

This script:
  1) Reads a text file from data/ground_truth/txt/<txt_file>.
  2) Concatenates that text below the standard prompt in src/prompts/llm_txt2csv/gpt-5.5.txt.
  3) Calls GPT-5.5 via the openai_api_text function, retrieving JSON output.
     - If the API call fails or returns invalid JSON, the script logs the error and exits.
  4) Converts the returned JSON to a single CSV in
     results_revisions/llm_txt2csv/gpt-5.5/<txt_stem>/temperature_<T>/run_<NN>/<txt_stem>.csv.
  5) Logs usage tokens and timing, storing a JSON run log in
     logs_revisions/llm_txt2csv/gpt-5.5/run_<timestamp>.json.
"""

import os
import sys
import re
import json
import time
import argparse
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Dict, List, Optional

from dotenv import load_dotenv

###############################################################################
# Project Paths
# These paths define where input data, prompts, results, and logs are stored.
# PROJECT_ROOT is computed relative to this script's location (two levels up).
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "ground_truth" / "txt"
# REVISION: Changed prompt filename from "gpt-4o.txt" to "gpt-5.5.txt"
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "llm_txt2csv" / "gpt-5.5.txt"
# REVISION: Changed from "results" to "results_revisions" and "gpt-4o" to "gpt-5.5"
RESULTS_DIR = PROJECT_ROOT / "results_revisions" / "llm_txt2csv" / "gpt-5.5"
# REVISION: Changed from "logs" to "logs_revisions" and "gpt-4o" to "gpt-5.5"
LOGS_DIR = PROJECT_ROOT / "logs_revisions" / "llm_txt2csv" / "gpt-5.5"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
# Reads the .env file to obtain the OpenAI API key needed for authentication.
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model constants
# REVISION: Changed from gpt-4o to gpt-5.5
MODEL_NAME = "gpt-5.5"                    # Short name for directory naming
# REVISION: Changed from "gpt-4o-2024-08-06" to "gpt-5.5"
FULL_MODEL_NAME = "gpt-5.5"               # Full GPT-5.5 model ID
# REVISION: Set to 65536 (maximum). Reasoning tokens count against this budget.
MAX_OUTPUT_TOKENS = 65536
SEED = 42                                 # Not used by OpenAI, kept for parity

###############################################################################
# Utility: Time formatting
# Converts raw seconds into a human-readable H:MM:SS string for log output.
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
# OpenAI GPT-5.5 API call for text-only usage
# Sends a text-only prompt (no image) to the OpenAI chat completions endpoint.
# Returns the model's text response along with token usage statistics.
###############################################################################
def openai_api_text(
    prompt: str,
    full_model_name: str,
    max_tokens: int,
    temperature: float,
    api_key: str
) -> tuple[Optional[str], dict]:
    """
    Call OpenAI's GPT-5.5 with text (no image).
    Returns (text_out, usage_info) or (None, {}).

    usage_info will be a dict with:
      {
        "prompt_tokens": <int>,
        "completion_tokens": <int>,
        "total_tokens": <int>
      }
    """

    # Set up HTTP headers with JSON content type and Bearer token authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Build the API request payload with the model name and text-only message
    # GPT-5.5 expects a "messages" list with role/user content for chat completions
    payload = {
        "model": full_model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
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
            raise ValueError(f"GPT-5.5 error {response.status_code}: {response.text}")
    except Exception as e:
        # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
        logging.error(f"GPT-5.5 call failed: {e}")
        return None, {}

###############################################################################
# Utility: Parse JSON from GPT-5.5 text response
# Extracts JSON from a code-fenced block (```json ... ```) if present,
# otherwise tries to parse the raw text directly as JSON.
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON if present; otherwise, fall back to raw text.
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
# Utility: Convert JSON to CSV
# Flattens a list of JSON objects (or a single object) into a CSV file.
# Columns are ordered with 'id' first (if present), then alphabetically.
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

    # Normalize input: wrap a single dict in a list for uniform handling
    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        # fallback: treat as single row with "value" column
        records = [{"value": str(json_data)}]

    # Collect the union of all keys across all records
    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    # Build fieldnames: 'id' first if it exists, then alphabetical for the rest
    fieldnames = []
    if "id" in all_keys:
        fieldnames.append("id")

    other_keys = [k for k in sorted(all_keys) if k != "id"]
    fieldnames.extend(other_keys)

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
# GPT-5.5 Text-Only API call (single attempt, no retry)
###############################################################################
def gpt55_api_call_text(prompt: str, temperature: float) -> Optional[dict]:
    """
    Call GPT-5.5 with text-only prompt (single attempt, no retry).

    Returns a dict:
      {
        "text": <the text response>,
        "usage": {
          "prompt_tokens": <int>,
          "completion_tokens": <int>,
          "total_tokens": <int>
        }
      }
    or None if the call fails or returns empty text.
    """
    text_out, usage_info = openai_api_text(
        prompt=prompt,
        full_model_name=FULL_MODEL_NAME,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        api_key=OPENAI_API_KEY
    )
    if not text_out:
        return None

    return {
        "text": text_out,
        "usage": {
            "prompt_tokens": usage_info.get("prompt_tokens", 0),
            "completion_tokens": usage_info.get("completion_tokens", 0),
            "total_tokens": usage_info.get("total_tokens", 0)
        }
    }

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the GPT-5.5 TXT -> JSON -> CSV pipeline.
    Orchestrates the full workflow: argument parsing, prompt loading,
    text concatenation, GPT-5.5 API call, JSON parsing, CSV generation,
    and logging.
    """
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    # --txt: the text filename located in data/ground_truth/txt/
    # --temperature: LLM sampling temperature (default 0.0 for deterministic output)
    # -------------------------------------------------------------------------
    # REVISION: Changed description from "GPT-4o" to "GPT-5.5"
    parser = argparse.ArgumentParser(description="GPT-5.5 TXT-to-JSON-to-CSV Pipeline")
    parser.add_argument(
        "--txt",
        required=True,
        help="Name of the TXT file in data/ground_truth/txt/, e.g. type-1.txt"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        # REVISION: Changed help text from "GPT-4o" to "GPT-5.5"
        help="LLM temperature for GPT-5.5 (default = 0.0)"
    )
    args = parser.parse_args()
    txt_name = args.txt
    temperature = args.temperature

    # Configure logging to stdout with timestamp and level prefix
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # REVISION: Changed pipeline banner from "GPT-4o" to "GPT-5.5"
    logging.info("=== GPT-5.5 TXT -> JSON -> CSV Pipeline ===")
    logging.info(f"TXT: {txt_name} | Temperature: {temperature} | Seed={SEED}")

    # -------------------------------------------------------------------------
    # Check for the input TXT file in the ground truth directory
    # -------------------------------------------------------------------------
    txt_path = DATA_DIR / txt_name
    if not txt_path.is_file():
        logging.error(f"TXT file not found at: {txt_path}")
        sys.exit(1)
    txt_stem = txt_path.stem

    # -------------------------------------------------------------------------
    # Load the GPT-5.5 prompt from the prompt file
    # The prompt contains instructions for how the model should parse
    # the text into structured JSON.
    # REVISION: Changed prompt filename from "gpt-4o.txt" to "gpt-5.5.txt"
    # -------------------------------------------------------------------------
    if not PROMPT_PATH.is_file():
        # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
        logging.error(f"Missing GPT-5.5 prompt file: {PROMPT_PATH}")
        sys.exit(1)

    base_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not base_prompt:
        logging.error(f"Prompt file is empty: {PROMPT_PATH}")
        sys.exit(1)
    # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
    logging.info(f"Loaded GPT-5.5 prompt from: {PROMPT_PATH}")

    # -------------------------------------------------------------------------
    # Combine the base prompt with the user's input text
    # The full prompt is: base_prompt + double newline + user text content
    # -------------------------------------------------------------------------
    user_text = txt_path.read_text(encoding="utf-8").strip()
    full_prompt = f"{base_prompt}\n\n{user_text}"

    # -------------------------------------------------------------------------
    # Prepare results folder
    # Directory structure:
    #   results_revisions/llm_txt2csv/gpt-5.5/<txt_stem>/temperature_X.X/run_nn/
    # Each run gets its own numbered subdirectory for reproducibility.
    # REVISION: Changed from results/ to results_revisions/ and gpt-4o to gpt-5.5
    # -------------------------------------------------------------------------
    txt_folder = RESULTS_DIR / txt_stem
    temp_folder = txt_folder / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Determine next run number by scanning for existing run_XX directories
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

    # Start timing the pipeline
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Single GPT-5.5 call (no retry)
    # Unlike image-based pipelines, this sends the entire text in one call.
    # REVISION: Changed function name from gpt4o_api_call_text to gpt55_api_call_text
    # -------------------------------------------------------------------------
    result = gpt55_api_call_text(prompt=full_prompt, temperature=temperature)
    if result is None:
        logging.error("GPT-5.5 API call failed.")
        sys.exit(1)

    response_text = result["text"]
    usage_meta = result["usage"]

    # Extract token usage from the API response
    prompt_tokens = usage_meta["prompt_tokens"]
    completion_tokens = usage_meta["completion_tokens"]
    total_tokens = usage_meta["total_tokens"]

    # REVISION: Changed log message from "GPT-4o" to "GPT-5.5"
    logging.info(
        f"GPT-5.5 usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
    )

    # -------------------------------------------------------------------------
    # JSON parsing (single attempt, no retry)
    # -------------------------------------------------------------------------
    try:
        parsed_data = parse_json_str(response_text)
    except ValueError as ve:
        logging.error(f"JSON parse error: {ve}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Save raw JSON output to a file in the run directory
    # This preserves the model's exact output for debugging and audit purposes.
    # -------------------------------------------------------------------------
    raw_json_path = run_folder / f"{txt_stem}.json"
    with raw_json_path.open("w", encoding="utf-8") as jf:
        json.dump(parsed_data, jf, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Optionally add "id" fields to each record
    # Ensures every record has a sequential integer ID for downstream use.
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
            return {"id": 1, "value": str(data)}

    final_data = add_ids_to_data(parsed_data)

    # -------------------------------------------------------------------------
    # Convert JSON -> CSV
    # Write the final structured data as a CSV file in the run directory.
    # -------------------------------------------------------------------------
    final_csv_path = run_folder / f"{txt_stem}.csv"
    convert_json_to_csv(final_data, final_csv_path)
    logging.info(f"Final CSV saved at: {final_csv_path}")

    # -------------------------------------------------------------------------
    # Log metadata
    # Write a JSON log file with run parameters, usage stats, and timing.
    # REVISION: Logs are written to logs_revisions/ instead of logs/
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
        "seed": SEED
    }

    with log_path.open("w", encoding="utf-8") as lf:
        json.dump(log_data, lf, indent=4)

    # -------------------------------------------------------------------------
    # Final summary
    # Print cumulative token usage and total pipeline duration to the console.
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Prompt tokens used: {prompt_tokens}")
    logging.info(f"Completion tokens used: {completion_tokens}")
    logging.info(f"Grand total tokens used: {total_tokens}")
    logging.info(f"Pipeline completed in {format_duration(total_duration)}.")
    logging.info("All done!")


if __name__ == "__main__":
    main()
