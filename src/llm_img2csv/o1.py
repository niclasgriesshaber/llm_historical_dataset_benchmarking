#!/usr/bin/env python3
"""
GPT-4o PDF -> PNG -> JSON -> CSV Pipeline (adapted for o1)

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls the o1 model for each page image, retrieving JSON output.
     - Automatically retries on any error (including JSON parsing failures).
     - Limits each page's retry attempts to RETRY_LIMIT_SECONDS; if no success in that window, it skips the page.
  3) Merges all returned JSON data into a single CSV (named <pdf_stem>.csv).
  4) Logs usage tokens per page (prompt/completion), accumulates them across all pages, and saves a JSON run log.

Comments still reference "GPT-4o" for clarity, but the actual API call is to "o1".
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
from pdf2image import convert_from_path
from PIL import Image

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

###############################################################################
# Model / Reasoning Config
###############################################################################
MODEL_NAME = "o1"                # Used in folder naming, doc references remain "GPT-4o"
FULL_MODEL_NAME = "o1"           # The actual model name we call
FOLDER_TEMPERATURE = "0.0"       # Hardcoded subfolder for naming only (not used in the API call)

# o1 reasoning-specific parameters
REASONING_CONFIG = {
    "reasoning_effort": "high",
    "max_completion_tokens": 100000
}

RETRY_LIMIT_SECONDS = 600  # shortened from 3600 to 600 seconds (10 min)
SEED = 42                  # For consistency (not used by OpenAI, but kept for reference)

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
# Utility: Parse JSON from the model's text response
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
        # If the response is something else, store it under a 'value' column
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
# o1 Call (replacing GPT-4o)
###############################################################################
def openai_o1_api(prompt: str, pil_image: Image.Image, api_key: str) -> (Optional[str], dict):
    """
    Call OpenAI’s o1 reasoning model with an image + text prompt.
    Returns (text_out, usage_info) or (None, {}).

    usage_info may have keys:
        {
          "prompt_tokens": ...,
          "completion_tokens": ...,
          "total_tokens": ...,
          "completion_tokens_details": {
             "reasoning_tokens": ...,
             "accepted_prediction_tokens": ...,
             "rejected_prediction_tokens": ...
          }
        }
    """
    import base64
    from io import BytesIO

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
        "model": FULL_MODEL_NAME,
        "reasoning_effort": REASONING_CONFIG["reasoning_effort"],
        "max_completion_tokens": REASONING_CONFIG["max_completion_tokens"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]
        # No temperature parameter
        # No seed parameter
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
            raise ValueError(f"OpenAI o1 error {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"OpenAI o1 call failed: {e}")
        return None, {}

###############################################################################
# Higher-level call wrapper with retry
###############################################################################
def o1_api_call_with_retry(prompt: str, pil_image: Image.Image) -> Optional[dict]:
    """
    Calls openai_o1_api with up to RETRY_LIMIT_SECONDS of retries.
    Returns {
      "text": <the text response>,
      "usage": <dict usage metadata>
    }
    or None if it fails.
    """
    start_retry = time.time()

    while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
        try:
            text_out, usage_info = openai_o1_api(
                prompt=prompt,
                pil_image=pil_image,
                api_key=OPENAI_API_KEY
            )
            if not text_out:
                logging.warning("o1 returned empty text; retrying...")
                continue

            return {"text": text_out, "usage": usage_info}

        except Exception as e:
            logging.warning(f"o1 API call failed: {e}. Retrying...")

    logging.error("o1 call did not succeed within the retry limit.")
    return None

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the GPT-4o (o1) PDF-to-JSON-to-CSV pipeline.
    """
    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="GPT-4o (o1) PDF -> JSON -> CSV Pipeline")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. my_file.pdf"
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
    continue_from_page = args.continue_from_page

    # -------------------------------------------------------------------------
    # Configure logging
    # -------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== GPT-4o PDF -> PNG -> JSON -> CSV Pipeline (o1) ===")
    logging.info(f"PDF: {pdf_name}")
    logging.info(f"Folder temperature label: {FOLDER_TEMPERATURE} (not used by the model).")
    logging.info(f"Seed (unused by o1, for reference): {SEED}")
    logging.info(f"Continue from page: {continue_from_page if continue_from_page else 'None (new run)'}")

    # -------------------------------------------------------------------------
    # Verify the PDF and environment
    # -------------------------------------------------------------------------
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY not set in .env or environment.")
        sys.exit(1)

    pdf_stem = Path(pdf_name).stem

    # -------------------------------------------------------------------------
    # Load the task prompt from: src/prompts/llm_img2csv/o1.txt (for consistency)
    # -------------------------------------------------------------------------
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Missing prompt file: {prompt_path}")
        sys.exit(1)

    task_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not task_prompt:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)

    logging.info(f"Loaded prompt from: {prompt_path}")

    # -------------------------------------------------------------------------
    # Create PNG (if needed) => data/page_by_page/PNG/<pdf_stem>
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
    # Prepare results folder => results/llm_img2csv/o1/<pdf_stem>/temperature_0.0/run_XX/
    # -------------------------------------------------------------------------
    pdf_folder = RESULTS_DIR / MODEL_NAME / pdf_stem
    temp_folder = pdf_folder / f"temperature_{FOLDER_TEMPERATURE}"
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

    # If no continue_from_page => new run folder
    if continue_from_page is None:
        next_run = highest_run_num + 1
        run_folder = temp_folder / f"run_{str(next_run).zfill(2)}"
        run_folder.mkdir(parents=True, exist_ok=False)
        logging.info(f"Creating new run folder: {run_folder}")
    else:
        # If user explicitly set --continue_from_page, reuse highest existing run folder
        if highest_run_num == 0:
            # no existing runs, but user wants to continue => create run_01
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

    run_page_json_dir = run_folder / "page_by_page"
    run_page_json_dir.mkdir(parents=True, exist_ok=True)

    final_csv_path = run_folder / f"{pdf_stem}.csv"

    # -------------------------------------------------------------------------
    # Accumulators for usage
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    # Extended usage
    total_reasoning_tokens = 0
    total_accepted_prediction_tokens = 0
    total_rejected_prediction_tokens = 0

    # -------------------------------------------------------------------------
    # Start timing
    # -------------------------------------------------------------------------
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Possibly skip ahead if --continue_from_page is set
    # -------------------------------------------------------------------------
    start_index = 1
    if continue_from_page is not None and continue_from_page > 1:
        start_index = continue_from_page
    # Adjust the PNG list to start from that page
    png_files = png_files[start_index - 1:]

    # -------------------------------------------------------------------------
    # Process each page
    # -------------------------------------------------------------------------
    for idx, png_path in enumerate(png_files, start=start_index):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        # Log image metadata
        try:
            with Image.open(png_path) as pil_image:
                width, height = pil_image.size
                dpi_value = pil_image.info.get("dpi", None)
                if dpi_value and len(dpi_value) == 2:
                    logging.info(f"Image metadata -> width={width}px, height={height}px, dpi={dpi_value}")
                else:
                    logging.info(f"Image metadata -> width={width}px, height={height}px, dpi=UNKNOWN")

                # Make the model call (o1) with retry
                result = o1_api_call_with_retry(task_prompt, pil_image)
        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            logging.info("")
            continue

        if result is None:
            logging.error(
                f"Skipping page {idx} because o1 API did not succeed within {RETRY_LIMIT_SECONDS} seconds."
            )
            logging.info("")
            continue

        response_text = result["text"]
        usage_meta = result["usage"]

        # ---------------------------------------------------------------------
        # Update usage counters for this page
        # ---------------------------------------------------------------------
        page_prompt_tokens = usage_meta.get("prompt_tokens", 0)
        page_completion_tokens = usage_meta.get("completion_tokens", 0)
        page_total_tokens = usage_meta.get("total_tokens", page_prompt_tokens + page_completion_tokens)

        total_prompt_tokens += page_prompt_tokens
        total_completion_tokens += page_completion_tokens
        total_tokens += page_total_tokens

        # Look for extended usage fields
        ctd = usage_meta.get("completion_tokens_details", {})
        page_reasoning = ctd.get("reasoning_tokens", 0)
        page_accepted = ctd.get("accepted_prediction_tokens", 0)
        page_rejected = ctd.get("rejected_prediction_tokens", 0)

        total_reasoning_tokens += page_reasoning
        total_accepted_prediction_tokens += page_accepted
        total_rejected_prediction_tokens += page_rejected

        logging.info(
            f"o1 usage for page {idx}: "
            f"prompt={page_prompt_tokens}, completion={page_completion_tokens}, total={page_total_tokens}, "
            f"reasoning={page_reasoning}, accepted_pred={page_accepted}, rejected_pred={page_rejected}"
        )
        logging.info(
            f"Accumulated usage so far: prompt={total_prompt_tokens}, "
            f"completion={total_completion_tokens}, total={total_tokens}, "
            f"reasoning={total_reasoning_tokens}, accepted_pred={total_accepted_prediction_tokens}, "
            f"rejected_pred={total_rejected_prediction_tokens}"
        )

        # ---------------------------------------------------------------------
        # JSON parsing (retry if needed)
        # ---------------------------------------------------------------------
        parse_start_time = time.time()
        parsed = None
        while True:
            try:
                parsed = parse_json_str(response_text)
            except ValueError as ve:
                if (time.time() - parse_start_time) > RETRY_LIMIT_SECONDS:
                    logging.error(
                        f"Skipping page {idx}: JSON parse still failing after {RETRY_LIMIT_SECONDS} seconds."
                    )
                    break
                logging.error(f"JSON parse error for page {idx}: {ve}")
                logging.error("Retrying the model call for JSON parse fix...")

                # Attempt a new model call
                try:
                    with Image.open(png_path) as pil_image_again:
                        new_result = o1_api_call_with_retry(task_prompt, pil_image_again)
                except Exception as e2:
                    logging.error(f"Failed to re-open image {png_path}: {e2}")
                    break

                if not new_result:
                    logging.error(
                        f"Could not fix JSON parse for page {idx}, skipping after {RETRY_LIMIT_SECONDS} seconds."
                    )
                    break

                # New text
                response_text = new_result["text"]
                usage_retry = new_result["usage"]

                # Accumulate usage again
                r_prompt = usage_retry.get("prompt_tokens", 0)
                r_completion = usage_retry.get("completion_tokens", 0)
                r_total = usage_retry.get("total_tokens", r_prompt + r_completion)

                total_prompt_tokens += r_prompt
                total_completion_tokens += r_completion
                total_tokens += r_total

                # Extended usage
                ctd_retry = usage_retry.get("completion_tokens_details", {})
                r_reasoning = ctd_retry.get("reasoning_tokens", 0)
                r_accepted = ctd_retry.get("accepted_prediction_tokens", 0)
                r_rejected = ctd_retry.get("rejected_prediction_tokens", 0)

                total_reasoning_tokens += r_reasoning
                total_accepted_prediction_tokens += r_accepted
                total_rejected_prediction_tokens += r_rejected

                logging.info(
                    f"[Retry usage] Additional tokens: prompt={r_prompt}, completion={r_completion}, total={r_total}"
                )
                logging.info(
                    f"Now accumulated: prompt={total_prompt_tokens}, completion={total_completion_tokens}, total={total_tokens}, "
                    f"reasoning={total_reasoning_tokens}, accepted_pred={total_accepted_prediction_tokens}, "
                    f"rejected_pred={total_rejected_prediction_tokens}"
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
    # Gather all JSON => single CSV
    # -------------------------------------------------------------------------
    logging.info("Gathering all JSON files from page_by_page folder to build final CSV...")
    all_page_json_files = sorted(run_page_json_dir.glob("page_*.json"))

    merged_data: List[Any] = []

    for fpath in all_page_json_files:
        page_str = fpath.stem.split("_")[1]
        page_num = int(page_str)

        with fpath.open("r", encoding="utf-8") as jf:
            content = json.load(jf)

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

    # Add an "id" field to each record for clarity
    for i, record in enumerate(merged_data, start=1):
        if isinstance(record, dict):
            record["id"] = i
        else:
            merged_data[i - 1] = {"id": i, "value": str(record)}

    convert_json_to_csv(merged_data, final_csv_path)
    logging.info(f"Final CSV saved at: {final_csv_path}")

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
        "reasoning_config": REASONING_CONFIG,
        "folder_temperature": FOLDER_TEMPERATURE,
        "run_directory": str(run_folder),
        "prompt_file": str(prompt_path),
        "pages_count": total_pages,
        "page_json_directory": str(run_page_json_dir),
        "final_csv": str(final_csv_path),
        "seed": SEED,
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": total_reasoning_tokens,
            "accepted_prediction_tokens": total_accepted_prediction_tokens,
            "rejected_prediction_tokens": total_rejected_prediction_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration)
    }

    with log_path.open("w", encoding="utf-8") as lf:
        json.dump(log_data, lf, indent=4)

    # -------------------------------------------------------------------------
    # Final usage summary
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary (including reasoning details) ===")
    logging.info(f"Total prompt tokens:               {total_prompt_tokens}")
    logging.info(f"Total completion tokens:           {total_completion_tokens}")
    logging.info(f"Total reasoning tokens:            {total_reasoning_tokens}")
    logging.info(f"Total accepted prediction tokens:   {total_accepted_prediction_tokens}")
    logging.info(f"Total rejected prediction tokens:   {total_rejected_prediction_tokens}")
    logging.info(f"Grand total of all tokens used:     {total_tokens}")

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()