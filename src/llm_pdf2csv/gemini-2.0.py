#!/usr/bin/env python3
"""
Gemini-2.0 PDF -> JSON -> CSV (Single Request) Pipeline

Usage:
  python3 gemini-2.0.py --pdf <your_pdf_name.pdf>

This script:
1. Loads a PDF from the 'data/pdfs/' directory.
2. Reads a corresponding prompt from 'src/prompts/llm_pdf2csv/gemini-2.0.txt'.
3. Calls the Gemini-2.0 LLM (via google.generativeai) to produce structured JSON.
4. Parses and cleans that JSON into final CSV output.
5. Appends an 'id' field to the CSV, with ascending integers starting at 1.
6. Saves artifacts (JSON, CSV, logs) in a dedicated run folder under 'results/llm_pdf2csv'.
7. Logs usage data (token counts, total runtime, etc.) for reproducibility.
"""

import os
import sys
import re
import json
import time
import argparse
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Dict, List, Optional

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_pdf2csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_pdf2csv"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# If you want logs also in the same run folder, you can remove this or just keep it.
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_pdf2csv"

###############################################################################
# Gemini Cookbook Imports (EXACT)
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

###############################################################################
# Model constants
###############################################################################
MODEL_NAME = "gemini-2.0"             # Short name for folder naming
FULL_MODEL_NAME = "gemini-2.0-flash"  # If you want an exact model variant
MAX_OUTPUT_TOKENS = 8192

# Retry limit in seconds (e.g., 1 hour = 3600).
# Adjust to 600 if you truly only want a 10-minute limit, etc.
RETRY_LIMIT_SECONDS = 3600

###############################################################################
# Utility: Time formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a time duration in seconds into a standard 'H:MM:SS' string for clearer logging output.

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: Formatted duration as 'HH:MM:SS'.
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
    Extracts JSON content from a Gemini-2.0 text response, looking for a fenced code block
    with triple backticks (``` or ```json). If no fenced block is found, the entire response
    is used (with backticks removed).

    It then parses the extracted content as JSON and returns the parsed object.

    Args:
        response_text (str): The text returned by Gemini-2.0.

    Returns:
        Any: The parsed JSON object (list, dict, etc.).

    Raises:
        ValueError: If the extracted JSON cannot be parsed.
    """
    fenced_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        response_text,
        re.IGNORECASE,
    )
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        candidate = response_text.strip().strip("```")

    return json.loads(candidate)

###############################################################################
# Utility: Reorder dictionary keys with page_number at the end (if desired)
###############################################################################
def reorder_dict_with_page_number(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a new dictionary ensuring 'page_number' and 'additional_information'
    appear last if present. This helps maintain a clean and consistent
    column ordering when converted to CSV.

    Args:
        d (Dict[str, Any]): The original dictionary.

    Returns:
        Dict[str, Any]: A new dictionary with keys sorted and 'page_number'
                        and 'additional_information' (if any) placed at the end.
    """
    special_keys = {"page_number", "additional_information"}
    base_keys = [k for k in d.keys() if k not in special_keys]
    base_keys.sort()

    out = {}
    for k in base_keys:
        out[k] = d[k]

    if "page_number" in d:
        out["page_number"] = d["page_number"]

    if "additional_information" in d:
        out["additional_information"] = d["additional_information"]

    return out

###############################################################################
# Utility: Convert JSON to CSV (with 'id' added as the last column)
###############################################################################
def convert_json_to_csv(json_data: Union[Dict, List], csv_path: Path) -> None:
    """
    Converts JSON data (either a single dictionary or a list of dictionaries)
    into a CSV file and saves it to csv_path.

    1) If the top-level object is a dictionary, it becomes one row.
    2) If the top-level object is a list of dictionaries, each element becomes a row.
    3) Reorders columns so that 'page_number' and 'additional_information' appear last.
    4) Adds an 'id' column at the end, starting from 1 and incrementing by 1 for each row.

    Args:
        json_data (Union[Dict, List]): Parsed JSON data (single dict or list of dicts).
        csv_path (Path): Destination file path for the CSV.
    """
    import csv

    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        # Fallback: treat as a single row with a "value" column
        records = [{"value": str(json_data)}]

    # Gather all keys from all records
    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    special_order = ["page_number", "additional_information"]
    base_keys = [k for k in all_keys if k not in special_order]
    base_keys.sort()
    # Add any special keys (in that order), then 'id' last.
    ordered_special = [k for k in special_order if k in all_keys]
    fieldnames = base_keys + ordered_special + ["id"]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row_id = 1
        for rec in records:
            if not isinstance(rec, dict):
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                row_data["id"] = row_id
                writer.writerow(row_data)
                row_id += 1
                continue

            row_data = {}
            for fn in fieldnames:
                if fn == "id":
                    row_data[fn] = row_id
                else:
                    row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)
            row_id += 1

###############################################################################
# EXACT Gemini Cookbook Code + Our Logic for a Single PDF Request (with retries)
###############################################################################
def gemini_api_call_for_pdf(prompt: str, pdf_path: Path, temperature: float = 0.0) -> Optional[dict]:
    """
    Sends a PDF and a text prompt to the Gemini-2.0 LLM (via google.generativeai)
    and retrieves a text response, along with token usage metadata. It retries
    calls for up to RETRY_LIMIT_SECONDS in case of errors or empty responses.

    Args:
        prompt (str): The text prompt/instruction for Gemini-2.0.
        pdf_path (Path): Path to the PDF file to be uploaded.
        temperature (float, optional): The LLM temperature for controlling
                                       response randomness.

    Returns:
        Optional[dict]: A dictionary with structure:
            {
                "text": <Gemini-2.0 response text>,
                "usage": {
                    "prompt_token_count": <int or None>,
                    "candidates_token_count": <int or None>,
                    "total_token_count": <int or None>,
                }
            }
        or None if the call fails after retries.
    """
    model = genai.GenerativeModel(model_name=FULL_MODEL_NAME)

    start_retry = time.time()
    while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
        try:
            # Upload file per the EXACT Gemini cookbook approach
            file_ref = genai.upload_file(str(pdf_path))

            # Token counting
            model.count_tokens([file_ref, prompt])

            # Generate content from the model
            response = model.generate_content(
                [file_ref, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )
            )

            if not response:
                logging.warning("Gemini-2.0 returned an empty response; retrying...")
                continue

            text_candidate = response.text
            if not text_candidate:
                logging.warning("Gemini-2.0 returned no text in the response; retrying...")
                continue

            # Gather usage metadata if available
            usage = {
                "prompt_token_count": getattr(response, "prompt_token_count", None),
                "candidates_token_count": getattr(response, "candidates_token_count", None),
                "total_token_count": getattr(response, "total_token_count", None),
            }

            return {
                "text": text_candidate,
                "usage": usage
            }

        except Exception as e:
            logging.warning(f"Gemini-2.0 call failed: {e}. Retrying...")

    logging.error("Gemini-2.0 call did not succeed within the retry time limit.")
    return None

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the Gemini-2.0 PDF->JSON->CSV (Single Request) pipeline.
    1) Parses CLI arguments for PDF name (and optional temperature).
    2) Sets up logging.
    3) Validates file paths (PDF, prompt).
    4) Calls Gemini-2.0 in one go with the entire PDF + prompt.
    5) Parses the LLM response into JSON, retrying if malformed.
    6) Saves JSON & CSV in a new run folder under 'results/llm_pdf2csv/gemini-2.0/<pdf_name>/temperature_...'.
    7) Logs final usage, runtime, and paths to a JSON log file.
    """
    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Gemini-2.0 PDF->JSON->CSV Single-Request Pipeline")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. my_file.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="(Currently not used in the EXACT snippet, but kept for continuity)"
    )
    args = parser.parse_args()
    pdf_name = args.pdf
    temperature = args.temperature

    # -------------------------------------------------------------------------
    # Configure logging
    # -------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== Gemini-2.0 Single-Request PDF -> JSON -> CSV Pipeline (Cookbook) ===")
    logging.info(f"PDF: {pdf_name} | Temperature: {temperature}")

    # -------------------------------------------------------------------------
    # Verify PDF
    # -------------------------------------------------------------------------
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    pdf_stem = pdf_path.stem

    # -------------------------------------------------------------------------
    # Load the PDF->JSON instructions (prompt)
    # -------------------------------------------------------------------------
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Missing prompt file: {prompt_path}")
        sys.exit(1)

    task_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not task_prompt:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)

    logging.info(f"Loaded Gemini-2.0 prompt from: {prompt_path}")

    # -------------------------------------------------------------------------
    # Prepare results folder => results/llm_pdf2csv/gemini-2.0/<pdf_name>/temperature_x.x/run_nn/
    # -------------------------------------------------------------------------
    pdf_folder = RESULTS_DIR / MODEL_NAME / pdf_name
    temp_folder = pdf_folder / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Identify next run_xx
    existing_runs = []
    for child in temp_folder.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                run_num = int(child.name.split("_")[1])
                existing_runs.append(run_num)
            except ValueError:
                pass
    next_run = max(existing_runs) + 1 if existing_runs else 1

    run_folder = temp_folder / f"run_{str(next_run).zfill(2)}"
    run_folder.mkdir(parents=True, exist_ok=False)
    logging.info(f"Created new run folder: {run_folder}")

    # We will place both JSON and CSV in this same folder:
    final_csv_path = run_folder / f"{pdf_stem}.csv"
    pdf_json_path = run_folder / f"{pdf_stem}.json"

    # -------------------------------------------------------------------------
    # Start pipeline timer
    # -------------------------------------------------------------------------
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Call Gemini in one request (with EXACT snippet usage).
    # -------------------------------------------------------------------------
    logging.info("Sending entire PDF to Gemini-2.0 in one request (cookbook style)...")
    result = gemini_api_call_for_pdf(prompt=task_prompt, pdf_path=pdf_path, temperature=temperature)

    if result is None:
        logging.error("Gemini-2.0 API did not succeed in the allowed time. Exiting.")
        sys.exit(1)

    response_text = result["text"]
    usage_meta = result["usage"] or {}

    # Summation counters for usage
    total_prompt_tokens = usage_meta.get("prompt_token_count", 0)
    total_candidates_tokens = usage_meta.get("candidates_token_count", 0)
    total_tokens = usage_meta.get("total_token_count", 0)

    logging.info(
        f"Gemini usage: input={total_prompt_tokens}, "
        f"candidate={total_candidates_tokens}, total={total_tokens}"
    )

    # -------------------------------------------------------------------------
    # Parse JSON with up to 1-hour retry
    # -------------------------------------------------------------------------
    parse_start_time = time.time()
    parsed_data = None

    while True:
        try:
            parsed_data = parse_json_str(response_text)
        except ValueError as ve:
            if (time.time() - parse_start_time) > RETRY_LIMIT_SECONDS:
                logging.error("JSON parse still failing after 1 hour. Exiting.")
                sys.exit(1)

            logging.error(f"JSON parse error: {ve}")
            logging.error("Will attempt a new Gemini call to fix JSON structure...")

            retry_result = gemini_api_call_for_pdf(prompt=task_prompt, pdf_path=pdf_path)
            if not retry_result:
                logging.error("No luck fixing JSON parse after 1 hour. Exiting.")
                sys.exit(1)

            rusage = retry_result["usage"] or {}
            total_prompt_tokens += rusage.get("prompt_token_count", 0)
            total_candidates_tokens += rusage.get("candidates_token_count", 0)
            total_tokens += rusage.get("total_token_count", 0)

            logging.info(
                f"[Retry usage] Accumulated tokens => "
                f"prompt={total_prompt_tokens}, candidates={total_candidates_tokens}, total={total_tokens}"
            )

            response_text = retry_result["text"]
        else:
            break

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------
    with pdf_json_path.open("w", encoding="utf-8") as jf:
        json.dump(parsed_data, jf, indent=2, ensure_ascii=False)
    logging.info(f"Saved JSON to {pdf_json_path}")

    # -------------------------------------------------------------------------
    # Reorder + Convert to CSV
    # -------------------------------------------------------------------------
    if isinstance(parsed_data, list):
        final_data = [
            reorder_dict_with_page_number(obj) if isinstance(obj, dict) else obj
            for obj in parsed_data
        ]
    elif isinstance(parsed_data, dict):
        final_data = reorder_dict_with_page_number(parsed_data)
    else:
        final_data = parsed_data

    convert_json_to_csv(final_data, final_csv_path)
    logging.info(f"CSV saved at {final_csv_path}")

    # -------------------------------------------------------------------------
    # Write a run log (JSON) in the same folder
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start_time
    log_path = LOGS_DIR / MODEL_NAME / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature_passed_in": temperature,
        "run_directory": str(run_folder),
        "prompt_file": str(prompt_path),
        "json_file": str(pdf_json_path),
        "csv_file": str(final_csv_path),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidates_tokens": total_candidates_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }
    with log_path.open("w", encoding="utf-8") as lf:
        json.dump(log_data, lf, indent=4)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logging.info(
        f"Pipeline completed in {format_duration(total_duration)} (H:MM:SS). All done!"
    )

if __name__ == "__main__":
    main()