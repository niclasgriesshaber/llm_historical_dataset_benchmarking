#!/usr/bin/env python3
"""
GPT-4o TXT -> JSON -> CSV Pipeline

This script:
  1) Reads a text file from data/ground_truth/txt/<txt_file>.
  2) Concatenates that text below the standard prompt in src/prompts/llm_txt2csv/gpt-4o.txt.
  3) Calls GPT-4o via the openai_api function, retrieving JSON output.
     - Retries up to 1 hour on any error (including JSON parse errors).
  4) Converts the returned JSON to a single CSV in
     results/llm_txt2csv/gpt-4o/<txt_stem>/temperature_<T>/run_<NN>/<txt_stem>.csv.
  5) Logs usage tokens and timing, storing a JSON run log in
     logs/llm_txt2csv/gpt-4o/run_<timestamp>.json.
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
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "ground_truth" / "txt"
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "llm_txt2csv" / "gpt-4o.txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_txt2csv" / "gpt-4o"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_txt2csv" / "gpt-4o"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model constants
MODEL_NAME = "gpt-4o"                   # Short name for directory naming
FULL_MODEL_NAME = "gpt-4o-2024-08-06"   # Full GPT-4o model ID
MAX_OUTPUT_TOKENS = 16134               # "max_tokens" in GPT-4o calls
RETRY_LIMIT_SECONDS = 3600              # up to 1 hour
SEED = 42                                # Not used by OpenAI, kept for parity

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
# "Precious" OpenAI GPT-4o API call for text-only usage
###############################################################################
def openai_api_text(
    prompt: str,
    full_model_name: str,
    max_tokens: int,
    temperature: float,
    api_key: str
) -> tuple[Optional[str], dict]:
    """
    Call OpenAI's GPT-4o with text (no image).
    Returns (text_out, usage_info) or (None, {}).

    usage_info will be a dict with:
      {
        "prompt_tokens": <int>,
        "completion_tokens": <int>,
        "total_tokens": <int>
      }
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # GPT-4o expects a "messages" list with role/user content for chat completions
    payload = {
        "model": full_model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            data = response.json()
            text_out = data["choices"][0]["message"]["content"]
            usage_info = data.get("usage", {})
            return text_out, usage_info
        else:
            raise ValueError(f"GPT-4o error {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"GPT-4o call failed: {e}")
        return None, {}

###############################################################################
# Utility: Parse JSON from GPT-4o text response
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON if present; otherwise, fall back to raw text.
    Then parse it as JSON. Raises ValueError if parsing fails.
    """
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
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

    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    # If "id" exists, we put it first, then alphabetical for the rest
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
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                writer.writerow(row_data)
                continue

            row_data = {}
            for fn in fieldnames:
                row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)

###############################################################################
# GPT-4o Text-Only API call with up to 1-hour retry
###############################################################################
def gpt4o_api_call_text(prompt: str, temperature: float) -> Optional[dict]:
    """
    Call GPT-4o with text-only prompt, retrying up to RETRY_LIMIT_SECONDS if needed.

    Returns a dict:
      {
        "text": <the text response>,
        "usage": {
          "prompt_tokens": <int>,
          "completion_tokens": <int>,
          "total_tokens": <int>
        }
      }
    or None if it fails after RETRY_LIMIT_SECONDS.
    """
    start_retry = time.time()

    while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
        try:
            text_out, usage_info = openai_api_text(
                prompt=prompt,
                full_model_name=FULL_MODEL_NAME,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=temperature,
                api_key=OPENAI_API_KEY
            )
            if not text_out:
                logging.warning("GPT-4o returned empty text; retrying...")
                continue

            return {
                "text": text_out,
                "usage": {
                    "prompt_tokens": usage_info.get("prompt_tokens", 0),
                    "completion_tokens": usage_info.get("completion_tokens", 0),
                    "total_tokens": usage_info.get("total_tokens", 0)
                }
            }
        except Exception as e:
            logging.warning(f"GPT-4o call failed: {e}. Retrying...")

    logging.error("GPT-4o call did not succeed after 1 hour.")
    return None

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the GPT-4o TXT -> JSON -> CSV pipeline.
    """
    parser = argparse.ArgumentParser(description="GPT-4o TXT-to-JSON-to-CSV Pipeline")
    parser.add_argument(
        "--txt",
        required=True,
        help="Name of the TXT file in data/ground_truth/txt/, e.g. type-1.txt"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for GPT-4o (default = 0.0)"
    )
    args = parser.parse_args()
    txt_name = args.txt
    temperature = args.temperature

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== GPT-4o TXT -> JSON -> CSV Pipeline ===")
    logging.info(f"TXT: {txt_name} | Temperature: {temperature} | Seed={SEED}")

    # Check for TXT file
    txt_path = DATA_DIR / txt_name
    if not txt_path.is_file():
        logging.error(f"TXT file not found at: {txt_path}")
        sys.exit(1)
    txt_stem = txt_path.stem

    # Load the GPT-4o prompt
    if not PROMPT_PATH.is_file():
        logging.error(f"Missing GPT-4o prompt file: {PROMPT_PATH}")
        sys.exit(1)

    base_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not base_prompt:
        logging.error(f"Prompt file is empty: {PROMPT_PATH}")
        sys.exit(1)
    logging.info(f"Loaded GPT-4o prompt from: {PROMPT_PATH}")

    # Combine prompt + user text
    user_text = txt_path.read_text(encoding="utf-8").strip()
    full_prompt = f"{base_prompt}\n\n{user_text}"

    # Prepare results folder => results/llm_txt2csv/gpt-4o/<txt_stem>/temperature_X.X/run_nn/
    txt_folder = RESULTS_DIR / txt_stem
    temp_folder = txt_folder / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Determine next run number
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

    overall_start_time = time.time()

    # Single GPT-4o call with up to 1-hour retry
    result = gpt4o_api_call_text(prompt=full_prompt, temperature=temperature)
    if result is None:
        # Abort if call fails after 1 hour
        total_duration = time.time() - overall_start_time
        logging.error(f"GPT-4o call failed after 1 hour. Elapsed: {format_duration(total_duration)}")
        sys.exit(1)

    response_text = result["text"]
    usage_meta = result["usage"]

    # Extract usage
    prompt_tokens = usage_meta["prompt_tokens"]
    completion_tokens = usage_meta["completion_tokens"]
    total_tokens = usage_meta["total_tokens"]

    logging.info(
        f"GPT-4o usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
    )

    # Retry JSON parse up to 1 hour if needed
    parse_start_time = time.time()
    parsed_data = None
    while True:
        try:
            parsed_data = parse_json_str(response_text)
        except ValueError as ve:
            if (time.time() - parse_start_time) > RETRY_LIMIT_SECONDS:
                logging.error("Skipping due to JSON parse failure after 1 hour.")
                break

            logging.error(f"JSON parse error: {ve}")
            logging.error("Retrying GPT-4o call to fix parse issues...")

            new_result = gpt4o_api_call_text(prompt=full_prompt, temperature=temperature)
            if not new_result:
                logging.error("Could not fix JSON parse after another attempt. Exiting.")
                break

            response_text = new_result["text"]
            usage_retry = new_result["usage"]

            # Accumulate usage
            prompt_tokens += usage_retry.get("prompt_tokens", 0)
            completion_tokens += usage_retry.get("completion_tokens", 0)
            total_tokens += usage_retry.get("total_tokens", 0)

            logging.info(
                f"[Retry usage] New accumulative usage => "
                f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
            )
        else:
            break

    if not parsed_data:
        # If we never succeeded in parsing JSON
        total_duration = time.time() - overall_start_time
        logging.error(f"Final parse failure. Elapsed: {format_duration(total_duration)}")
        sys.exit(1)

    # Save raw JSON
    raw_json_path = run_folder / f"{txt_stem}.json"
    with raw_json_path.open("w", encoding="utf-8") as jf:
        json.dump(parsed_data, jf, indent=2, ensure_ascii=False)

    # Optionally add "id" fields
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

    # Convert JSON -> CSV
    final_csv_path = run_folder / f"{txt_stem}.csv"
    convert_json_to_csv(final_data, final_csv_path)
    logging.info(f"Final CSV saved at: {final_csv_path}")

    # Log metadata
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

    # Final summary
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Prompt tokens used: {prompt_tokens}")
    logging.info(f"Completion tokens used: {completion_tokens}")
    logging.info(f"Grand total tokens used: {total_tokens}")
    logging.info(f"Pipeline completed in {format_duration(total_duration)}.")
    logging.info("All done!")


if __name__ == "__main__":
    main()