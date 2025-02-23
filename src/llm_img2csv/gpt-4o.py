#!/usr/bin/env python3
"""
GPT-4o PDF -> PNG -> JSON -> CSV Pipeline

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls GPT-4o for each page image, retrieving JSON output.
     - Automatically retries on any error (including JSON parsing failures).
     - Limits each page's retry attempts to 1 hour; if no success within that hour, it skips the page.
  3) Merges all returned JSON data into a single CSV (named <pdf_stem>.csv).
  4) Logs usage tokens per page (prompt/completion), accumulates them across all pages, and saves a JSON run log.

References to GPT-4o in comments are intentional to maintain clarity.
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
# OpenAI GPT-4 (GPT-4o) call function
# NOTE: We do NOT modify the 'openai_api' function logic beyond what's necessary
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
    Call OpenAI’s GPT-4o with an image + text prompt.
    Returns (text_out, usage_info) or (None, {}).

    usage_info is a dict with keys: {prompt_tokens, completion_tokens, total_tokens}.
    """
    import base64
    from io import BytesIO

    # Ensure RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    with BytesIO() as buffer:
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

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
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": SEED
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=payload
        )
        if response.status_code == 200:
            data = response.json()
            text_out = data["choices"][0]["message"]["content"]
            usage_info = data.get("usage", {})
            return text_out, usage_info
        else:
            raise ValueError(f"OpenAI GPT-4o error {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"OpenAI GPT-4o call failed: {e}")
        return None, {}

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_img2csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_img2csv"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_img2csv"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model constants
MODEL_NAME = "gpt-4o"                    # Short name for folder naming
FULL_MODEL_NAME = "gpt-4o-2024-08-06"    # Full GPT-4o model ID
MAX_OUTPUT_TOKENS = 16134               # "max_tokens" in openai_api
RETRY_LIMIT_SECONDS = 600               # shortened from 3600 to 600 seconds (10 min)
SEED = 42                                # For consistency (not used by OpenAI)

###############################################################################
# Utility: Time formatting
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
# Utility: Parse JSON from GPT-4o text response
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON if present, otherwise fallback to raw text.
    Then parse it as JSON. Raises ValueError if parsing fails.
    """
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        candidate = response_text.strip().strip("`")
    return json.loads(candidate)

###############################################################################
# Utility: Reorder dictionary keys with page_number at the end
###############################################################################
def reorder_dict_with_page_number(d: Dict[str, Any], page_number: int) -> Dict[str, Any]:
    """
    Return a new dict that includes a 'page_number' key at the end
    and ensures 'additional_information' is last if present.
    """
    special_keys = {"page_number", "additional_information"}
    base_keys = [k for k in d.keys() if k not in special_keys]
    base_keys.sort()

    out = {}
    for k in base_keys:
        out[k] = d[k]

    out["page_number"] = page_number

    if "additional_information" in d:
        out["additional_information"] = d["additional_information"]

    return out

###############################################################################
# Utility: Convert JSON to CSV
###############################################################################
def convert_json_to_csv(json_data: Union[Dict, List], csv_path: Path) -> None:
    """
    Flatten JSON objects/arrays into a CSV at csv_path.
    1) If top-level is a single dict, that's 1 row.
    2) If top-level is a list, each element is a row.
    3) Reorder columns so 'page_number' & 'additional_information' come last.
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

    special_order = ["page_number", "additional_information"]
    base_keys = [k for k in all_keys if k not in special_order]
    base_keys.sort()
    fieldnames = base_keys + [k for k in special_order if k in all_keys]

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
# GPT-4o API Call with up to 10-minute retry (adjusted from 1-hour)
###############################################################################
def gpt4o_api_call(
    prompt: str,
    pil_image: Image.Image,
    temperature: float
) -> Optional[dict]:
    """
    Call GPT-4o with the given prompt + image, retry up to RETRY_LIMIT_SECONDS if needed.
    Returns a dict:
      {
        "text": <the text response>,
        "usage": <usage metadata with {prompt_tokens, completion_tokens, total_tokens}>
      }
    or None if it fails after RETRY_LIMIT_SECONDS.
    """
    start_retry = time.time()

    while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
        try:
            text_out, usage_info = openai_api(
                prompt=prompt,
                pil_image=pil_image,
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
                "usage": usage_info
            }

        except Exception as e:
            logging.warning(f"GPT-4o call failed: {e}. Retrying...")

    logging.error("GPT-4o call did not succeed within the retry limit.")
    return None

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the GPT-4o PDF-to-CSV pipeline.
    """
    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="GPT-4o PDF-to-JSON-to-CSV Pipeline")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. my_file.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for GPT-4o (default = 0.0)"
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
    # -------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== GPT-4o PDF -> PNG -> JSON -> CSV Pipeline ===")
    logging.info(f"PDF: {pdf_name} | Temperature: {temperature} | Seed={SEED}")
    logging.info(f"Continue from page: {continue_from_page if continue_from_page else 'None (create new run)'}")

    # -------------------------------------------------------------------------
    # Verify the PDF exists
    # -------------------------------------------------------------------------
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    pdf_stem = Path(pdf_name).stem

    # -------------------------------------------------------------------------
    # Load the task prompt for GPT-4o
    # -------------------------------------------------------------------------
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Missing prompt file: {prompt_path}")
        sys.exit(1)

    task_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not task_prompt:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)

    logging.info(f"Loaded GPT-4o prompt from: {prompt_path}")

    # -------------------------------------------------------------------------
    # Check / create data/page_by_page/PNG/<pdf_stem> for PNG generation
    # -------------------------------------------------------------------------
    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"No PNG folder found; converting PDF -> PNG in {png_dir} ...")
        png_dir.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(pdf_path))
        for i, pil_img in enumerate(pages, start=1):
            out_png = png_dir / f"page_{i:04d}.png"
            pil_img.save(out_png, "PNG")
        logging.info(f"Created {len(pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"Folder {png_dir} already exists; skipping PDF->PNG step.")

    # Gather PNG files
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error(f"No PNG pages found in {png_dir}. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)

    # -------------------------------------------------------------------------
    # Prepare results folder => results/llm_img2csv/gpt-4o/<pdf_name>/temperature_x.x/run_nn/
    # -------------------------------------------------------------------------
    pdf_folder = RESULTS_DIR / MODEL_NAME / pdf_name.split(".")[0]
    temp_folder = pdf_folder / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

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

    # Subfolder for page-level JSON
    run_page_json_dir = run_folder / "page_by_page"
    run_page_json_dir.mkdir(parents=True, exist_ok=True)

    # Final CSV path -> run_folder/<pdf_stem>.csv
    final_csv_path = run_folder / f"{pdf_stem}.csv"

    # -------------------------------------------------------------------------
    # Accumulators for usage tokens only
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # -------------------------------------------------------------------------
    # Start timing the entire pipeline
    # -------------------------------------------------------------------------
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Process each page image from `continue_from_page` onward if it was given
    # -------------------------------------------------------------------------
    start_index = 1
    if continue_from_page is not None and continue_from_page > 1:
        start_index = continue_from_page
    # Adjust the png_files to start from that page index
    png_files = png_files[start_index - 1:]

    for idx, png_path in enumerate(png_files, start=start_index):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        # Log image metadata before calling GPT-4o
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

                # GPT-4o call with up to RETRY_LIMIT_SECONDS
                result = gpt4o_api_call(
                    prompt=task_prompt,
                    pil_image=pil_image,
                    temperature=temperature
                )
        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            logging.info("")
            continue

        if result is None:
            logging.error(
                f"Skipping page {idx} because GPT-4o API did not succeed within {RETRY_LIMIT_SECONDS} seconds."
            )
            logging.info("")
            continue

        response_text = result["text"]
        usage_meta = result["usage"]

        # ---------------------------------------------------------------------
        # Update usage accumulators for this page
        # ---------------------------------------------------------------------
        page_prompt_tokens = usage_meta.get("prompt_tokens", 0)
        page_completion_tokens = usage_meta.get("completion_tokens", 0)
        page_total_tokens = usage_meta.get("total_tokens", page_prompt_tokens + page_completion_tokens)

        total_prompt_tokens += page_prompt_tokens
        total_completion_tokens += page_completion_tokens
        total_tokens += page_total_tokens

        logging.info(
            f"GPT-4o usage for page {idx}: "
            f"input={page_prompt_tokens}, completion={page_completion_tokens}, total={page_total_tokens}"
        )
        logging.info(
            f"Accumulated usage so far: input={total_prompt_tokens}, "
            f"completion={total_completion_tokens}, total={total_tokens}"
        )

        # ---------------------------------------------------------------------
        # JSON parsing (retry up to RETRY_LIMIT_SECONDS)
        # ---------------------------------------------------------------------
        parse_start_time = time.time()
        while True:
            try:
                parsed = parse_json_str(response_text)
            except ValueError as ve:
                if (time.time() - parse_start_time) > RETRY_LIMIT_SECONDS:
                    logging.error(
                        f"Skipping page {idx}: JSON parse still failing after {RETRY_LIMIT_SECONDS} seconds."
                    )
                    parsed = None
                    break
                logging.error(f"JSON parse error for page {idx}: {ve}")
                logging.error("Retrying GPT-4o call for JSON parse fix...")

                # Attempt a new GPT-4o call
                try:
                    with Image.open(png_path) as pil_image_again:
                        new_result = gpt4o_api_call(
                            prompt=task_prompt,
                            pil_image=pil_image_again,
                            temperature=temperature
                        )
                except Exception as e2:
                    logging.error(f"Failed to re-open image {png_path}: {e2}")
                    parsed = None
                    break

                if not new_result:
                    logging.error(
                        f"Could not fix JSON parse for page {idx}, skipping after {RETRY_LIMIT_SECONDS} seconds."
                    )
                    parsed = None
                    break

                response_text = new_result["text"]
                usage_retry = new_result["usage"]

                # Accumulate usage again
                retry_ptc = usage_retry.get("prompt_tokens", 0)
                retry_ctc = usage_retry.get("completion_tokens", 0)
                retry_ttc = usage_retry.get("total_tokens", retry_ptc + retry_ctc)

                total_prompt_tokens += retry_ptc
                total_completion_tokens += retry_ctc
                total_tokens += retry_ttc

                logging.info(
                    f"[Retry usage] Additional tokens: input={retry_ptc}, "
                    f"completion={retry_ctc}, total={retry_ttc} | "
                    f"New accumulated: input={total_prompt_tokens}, "
                    f"completion={total_completion_tokens}, total={total_tokens}"
                )
            else:
                # parse succeeded
                break

        if not parsed:
            logging.info("")
            continue

        # ---------------------------------------------------------------------
        # Save page-level JSON
        # ---------------------------------------------------------------------
        page_json_path = run_page_json_dir / f"{png_path.stem}.json"
        with page_json_path.open("w", encoding="utf-8") as jf:
            json.dump(parsed, jf, indent=2, ensure_ascii=False)

        # ---------------------------------------------------------------------
        # Timing / Estimation
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
    # -------------------------------------------------------------------------
    for i, record in enumerate(merged_data, start=1):
        if isinstance(record, dict):
            record["id"] = i
        else:
            merged_data[i-1] = {"id": i, "value": str(record)}

    # -------------------------------------------------------------------------
    # Convert merged_data to CSV
    # -------------------------------------------------------------------------
    convert_json_to_csv(merged_data, final_csv_path)
    logging.info(f"Final CSV (all pages) saved at: {final_csv_path}")

    # -------------------------------------------------------------------------
    # Write JSON log with run metadata
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
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Total input (prompt) tokens used: {total_prompt_tokens}")
    logging.info(f"Total completion tokens used: {total_completion_tokens}")
    logging.info(f"Grand total of all tokens used: {total_tokens}")

    # -------------------------------------------------------------------------
    # Final logging
    # -------------------------------------------------------------------------
    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()