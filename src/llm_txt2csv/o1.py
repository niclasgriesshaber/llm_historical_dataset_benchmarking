#!/usr/bin/env python3
"""
GPT-4o TXT -> JSON -> CSV Pipeline (adapted for o1)

This script:
  1) Reads a text file from data/ground_truth/txt/<txt_file>.
  2) Concatenates that text below the standard prompt in src/prompts/llm_txt2csv/o1.txt.
  3) Calls o1 (previously GPT-4o) via an openai_o1_api function, retrieving JSON output.
     - Retries up to 1 hour on any error (including JSON parse errors).
  4) Converts the returned JSON to a single CSV in
     results/llm_txt2csv/o1/<txt_stem>/temperature_0.0/run_<NN>/<txt_stem>.csv.
  5) Logs usage tokens and timing, storing a JSON run log in
     logs/llm_txt2csv/o1/run_<timestamp>.json.
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
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "llm_txt2csv" / "o1.txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_txt2csv" / "o1"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_txt2csv" / "o1"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model constants for o1
MODEL_NAME = "o1"  
FOLDER_TEMPERATURE = "0.0"  # Hardcoded for folder naming only
RETRY_LIMIT_SECONDS = 3600  # Up to 1 hour
SEED = 42                   # Not used by o1, just for reference

# Reasoning parameters for o1
REASONING_CONFIG = {
    "reasoning_effort": "high",
    "max_completion_tokens": 100000
}

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
# o1 API text call (replaces GPT-4o text-only usage)
###############################################################################
def openai_o1_api_text(prompt: str, api_key: str) -> (Optional[str], dict):
    """
    Call OpenAI’s o1 model with a text-only prompt (no image).
    Returns (text_out, usage_info) or (None, {}).

    usage_info may have:
      {
        "prompt_tokens": <int>,
        "completion_tokens": <int>,
        "total_tokens": <int>,
        "completion_tokens_details": {
            "reasoning_tokens": ...,
            "accepted_prediction_tokens": ...,
            "rejected_prediction_tokens": ...
        }
      }
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": MODEL_NAME,  # "o1"
        "reasoning_effort": REASONING_CONFIG["reasoning_effort"],
        "max_completion_tokens": REASONING_CONFIG["max_completion_tokens"],
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
        # No temperature
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
            raise ValueError(f"o1 error {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"o1 call failed: {e}")
        return None, {}

###############################################################################
# Utility: Parse JSON from text response
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
    3) Reorder columns so 'id' (if present) is near the front, with others alphabetical.
    """
    import csv

    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        records = [{"value": str(json_data)}]

    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    fieldnames = []
    # Put "id" first if it exists
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
                # turn it into a dict with a single "value" field
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                writer.writerow(row_data)
                continue

            row_data = {}
            for fn in fieldnames:
                row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)

###############################################################################
# o1 text-only API call with up to 1-hour retry
###############################################################################
def o1_api_call_text(prompt: str) -> Optional[dict]:
    """
    Calls openai_o1_api_text with up to RETRY_LIMIT_SECONDS of retries.
    Returns a dict:
      {
        "text": <the text response>,
        "usage": {
          "prompt_tokens": <int>,
          "completion_tokens": <int>,
          "total_tokens": <int>,
          "completion_tokens_details": {
             "reasoning_tokens": ...,
             "accepted_prediction_tokens": ...,
             "rejected_prediction_tokens": ...
          }
        }
      }
    or None if it fails after RETRY_LIMIT_SECONDS.
    """
    start_retry = time.time()

    while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
        try:
            text_out, usage_info = openai_o1_api_text(
                prompt=prompt,
                api_key=OPENAI_API_KEY
            )
            if not text_out:
                logging.warning("o1 returned empty text; retrying...")
                continue

            return {"text": text_out, "usage": usage_info}
        except Exception as e:
            logging.warning(f"o1 call failed: {e}. Retrying...")

    logging.error("o1 call did not succeed after 1 hour.")
    return None

###############################################################################
# Main
###############################################################################
def main():
    """
    1) Reads a .txt from data/ground_truth/txt/<txt_file>.
    2) Merges it with the prompt in src/prompts/llm_txt2csv/o1.txt.
    3) Calls o1 => receives JSON => writes CSV.
    4) Logs usage tokens, saving a JSON run log in logs/llm_txt2csv/o1/.
    """
    parser = argparse.ArgumentParser(description="o1 TXT->JSON->CSV Pipeline")
    parser.add_argument(
        "--txt",
        required=True,
        help="Name of the TXT file in data/ground_truth/txt/, e.g. type-1.txt"
    )
    args = parser.parse_args()
    txt_name = args.txt

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== GPT-4o TXT -> JSON -> CSV Pipeline (o1) ===")
    logging.info(f"TXT: {txt_name}, no temperature used. Seed={SEED}")

    # Check environment & text file
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY not set in .env or environment.")
        sys.exit(1)

    txt_path = DATA_DIR / txt_name
    if not txt_path.is_file():
        logging.error(f"TXT file not found at: {txt_path}")
        sys.exit(1)
    txt_stem = txt_path.stem

    # Load the base prompt from: src/prompts/llm_txt2csv/o1.txt
    if not PROMPT_PATH.is_file():
        logging.error(f"Missing prompt file: {PROMPT_PATH}")
        sys.exit(1)

    base_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not base_prompt:
        logging.error(f"Prompt file is empty: {PROMPT_PATH}")
        sys.exit(1)

    logging.info(f"Loaded base prompt from: {PROMPT_PATH}")

    # Merge user text with base prompt
    user_text = txt_path.read_text(encoding="utf-8").strip()
    full_prompt = f"{base_prompt}\n\n{user_text}"

    # Prepare results folder => results/llm_txt2csv/o1/<txt_stem>/temperature_0.0/run_NN/
    txt_folder = RESULTS_DIR / txt_stem
    temp_folder = txt_folder / f"temperature_{FOLDER_TEMPERATURE}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Find existing run folders
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

    # Single o1 call with up to 1-hour retry
    result = o1_api_call_text(prompt=full_prompt)
    if result is None:
        total_duration = time.time() - overall_start_time
        logging.error(f"o1 call failed after 1 hour. Elapsed: {format_duration(total_duration)}")
        sys.exit(1)

    response_text = result["text"]
    usage_meta = result["usage"]

    # Basic usage fields
    prompt_tokens = usage_meta.get("prompt_tokens", 0)
    completion_tokens = usage_meta.get("completion_tokens", 0)
    total_tokens = usage_meta.get("total_tokens", 0)

    # Extended usage fields
    ctd = usage_meta.get("completion_tokens_details", {})
    reasoning_tokens = ctd.get("reasoning_tokens", 0)
    accepted_prediction_tokens = ctd.get("accepted_prediction_tokens", 0)
    rejected_prediction_tokens = ctd.get("rejected_prediction_tokens", 0)

    logging.info(
        f"o1 usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, "
        f"reasoning={reasoning_tokens}, accepted_pred={accepted_prediction_tokens}, "
        f"rejected_pred={rejected_prediction_tokens}"
    )

    # JSON parse (retry if needed)
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
            logging.error("Retrying o1 call to fix parse issues...")

            new_result = o1_api_call_text(prompt=full_prompt)
            if not new_result:
                logging.error("Could not fix JSON parse after another attempt. Exiting.")
                break

            response_text = new_result["text"]
            usage_retry = new_result["usage"]

            # Add usage from the retry
            prompt_tokens += usage_retry.get("prompt_tokens", 0)
            completion_tokens += usage_retry.get("completion_tokens", 0)
            total_tokens += usage_retry.get("total_tokens", 0)

            ctd_retry = usage_retry.get("completion_tokens_details", {})
            reasoning_tokens += ctd_retry.get("reasoning_tokens", 0)
            accepted_prediction_tokens += ctd_retry.get("accepted_prediction_tokens", 0)
            rejected_prediction_tokens += ctd_retry.get("rejected_prediction_tokens", 0)

            logging.info(
                f"[Retry usage] New accumulative usage => "
                f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, "
                f"reasoning={reasoning_tokens}, accepted_pred={accepted_prediction_tokens}, "
                f"rejected_pred={rejected_prediction_tokens}"
            )
        else:
            break

    if not parsed_data:
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
        "reasoning_config": REASONING_CONFIG,
        "folder_temperature": FOLDER_TEMPERATURE,
        "run_directory": str(run_folder),
        "prompt_file": str(PROMPT_PATH),
        "final_csv": str(final_csv_path),
        "json_output": str(raw_json_path),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "accepted_prediction_tokens": accepted_prediction_tokens,
            "rejected_prediction_tokens": rejected_prediction_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
        "seed": SEED
    }

    with log_path.open("w", encoding='utf-8') as lf:
        json.dump(log_data, lf, indent=4)

    # Final summary
    logging.info("=== Final Usage Summary (including new reasoning fields) ===")
    logging.info(f"Prompt tokens used:            {prompt_tokens}")
    logging.info(f"Completion tokens used:        {completion_tokens}")
    logging.info(f"Reasoning tokens used:         {reasoning_tokens}")
    logging.info(f"Accepted prediction tokens:    {accepted_prediction_tokens}")
    logging.info(f"Rejected prediction tokens:    {rejected_prediction_tokens}")
    logging.info(f"Grand total tokens used:       {total_tokens}")
    logging.info(f"Pipeline completed in {format_duration(total_duration)}.")
    logging.info("All done!")


if __name__ == "__main__":
    main()