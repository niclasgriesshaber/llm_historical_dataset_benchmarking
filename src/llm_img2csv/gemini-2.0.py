#!/usr/bin/env python3
"""
Gemini-2.0 PDF -> PNG -> JSON -> CSV Pipeline

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls Gemini-2.0 for each page image, retrieving JSON output.
     - Automatically retries on any error (including JSON parsing failures).
     - Limits each page's retry attempts to 1 hour; if no success within that hour, it skips the page.
  3) Merges **all** returned JSON data (from the entire run_page_json_dir) into a single CSV (named <pdf_stem>.csv).
  4) Logs usage tokens per page, accumulates them across all pages, and saves a JSON run log.
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
from pdf2image import convert_from_path
from PIL import Image

# Google-GenAI (Gemini-2.0) library
import google.genai as genai
from google.genai import types

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
API_KEY = os.getenv("GOOGLE_API_KEY")

# Model constants
MODEL_NAME = "gemini-2.0"           # Short name for folder naming
FULL_MODEL_NAME = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 8192
RETRY_LIMIT_SECONDS = 600  # up to 10 max retries per page

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
# Utility: Parse JSON from Gemini-2.0 text response
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON if present, otherwise fallback to raw text.
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
        # fallback to entire response, removing any backticks
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
        # Fallback: treat as a single row with "value" column
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
                # Fallback: convert to dict with "value" column
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                writer.writerow(row_data)
                continue

            row_data = {}
            for fn in fieldnames:
                row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)

###############################################################################
# Gemini-2.0 API Call with up to 1-hour retry
###############################################################################
def gemini_api_call(
    prompt: str,
    pil_image: Image.Image,
    temperature: float
) -> Optional[dict]:
    """
    Call Gemini-2.0 with the given prompt + image, retry up to 1 hour if needed.
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
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_file = tmp.name
                pil_image.save(tmp_file, "PNG")

            file_upload = client.files.upload(path=tmp_file)

            response = client.models.generate_content(
                model=FULL_MODEL_NAME,
                contents=[
                    types.Part.from_uri(
                        file_uri=file_upload.uri,
                        mime_type=file_upload.mime_type,
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    seed=42,
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

        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass

    logging.error("Gemini-2.0 call did not succeed after 1 hour.")
    return None

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the Gemini-2.0 PDF-to-CSV pipeline.
    """
    # -------------------------------------------------------------------------
    # Parse arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Gemini-2.0 PDF-to-JSON-to-CSV Pipeline")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. my_file.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for Gemini-2.0 (default = 0.0)"
    )
    # Changed to default=None so we can detect if it's omitted vs. provided.
    parser.add_argument(
        "--continue_from_page",
        type=int,
        default=None,
        help="Page number to continue from (if omitted, always create a new run)."
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

    logging.info("=== Gemini-2.0 PDF -> PNG -> JSON -> CSV Pipeline ===")
    logging.info(f"PDF: {pdf_name} | Temperature: {temperature} | Continue from page: {continue_from_page}")

    # -------------------------------------------------------------------------
    # Verify the PDF exists
    # -------------------------------------------------------------------------
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    pdf_stem = Path(pdf_name).stem

    # -------------------------------------------------------------------------
    # Load the task prompt for Gemini-2.0
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
    # Prepare results folder => results/llm_img2csv/gemini-2.0/<pdf_name>/temperature_x.x/run_nn/
    # -------------------------------------------------------------------------
    pdf_folder = RESULTS_DIR / MODEL_NAME / pdf_name
    temp_folder = pdf_folder / f"temperature_{temperature}"
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

    # -------------------------------------------------------------------------
    # Run folder logic
    # -------------------------------------------------------------------------
    if continue_from_page is None:
        #
        # If --continue_from_page is omitted, always create a new run_x+1 folder
        #
        next_run = highest_run_num + 1
        run_folder = temp_folder / f"run_{str(next_run).zfill(2)}"
        run_folder.mkdir(parents=True, exist_ok=False)
        logging.info(f"No --continue_from_page given; created new run folder: {run_folder}")
    else:
        #
        # If --continue_from_page was provided, reuse or create the highest run
        #
        if highest_run_num == 0:
            # no runs exist => make run_01
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
    # Accumulators for usage tokens, data for newly processed pages
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0

    # -------------------------------------------------------------------------
    # Start timing the entire pipeline
    # -------------------------------------------------------------------------
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Process each page image from continue_from_page onward
    # -------------------------------------------------------------------------
    # If continue_from_page is None, we start from page 1.
    start_page = continue_from_page if continue_from_page is not None else 1
    if start_page > 1:
        # Slice the list to skip pages before 'start_page'.
        png_files = png_files[start_page - 1:]

    for idx, png_path in enumerate(png_files, start=start_page):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        # Log image metadata before calling Gemini-2.0
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

                # Gemini-2.0 call with up to 1-hour retry
                result = gemini_api_call(
                    prompt=task_prompt,
                    pil_image=pil_image,
                    temperature=temperature
                )
        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            continue

        if result is None:
            logging.error(
                f"Skipping page {idx} because Gemini-2.0 API did not succeed within 1 hour."
            )
            logging.info("")
            continue

        response_text = result["text"]
        usage_meta = result["usage"]

        # ---------------------------------------------------------------------
        # Update usage accumulators for this page
        # ---------------------------------------------------------------------
        page_prompt_tokens = usage_meta.prompt_token_count or 0
        page_candidate_tokens = usage_meta.candidates_token_count or 0
        page_total_tokens = usage_meta.total_token_count or 0

        total_prompt_tokens += page_prompt_tokens
        total_candidates_tokens += page_candidate_tokens
        total_tokens += page_total_tokens

        logging.info(
            f"Gemini-2.0 usage for page {idx}: "
            f"input={page_prompt_tokens}, candidate={page_candidate_tokens}, total={page_total_tokens}"
        )
        logging.info(
            f"Accumulated usage so far: input={total_prompt_tokens}, "
            f"candidate={total_candidates_tokens}, total={total_tokens}"
        )

        # ---------------------------------------------------------------------
        # JSON parsing (retry up to 1 hour)
        # ---------------------------------------------------------------------
        parse_start_time = time.time()
        while True:
            try:
                parsed = parse_json_str(response_text)
            except ValueError as ve:
                if (time.time() - parse_start_time) > RETRY_LIMIT_SECONDS:
                    logging.error(
                        f"Skipping page {idx}: JSON parse still failing after 1 hour."
                    )
                    parsed = None
                    break
                logging.error(f"JSON parse error for page {idx}: {ve}")
                logging.info(f"Current response text: \n{response_text}\n")
                logging.error("Retrying Gemini-2.0 call for JSON parse fix...")
                new_result = gemini_api_call(
                    prompt=task_prompt,
                    pil_image=Image.open(png_path),
                    temperature=temperature
                )
                if not new_result:
                    logging.error(
                        f"Could not fix JSON parse for page {idx}, skipping after 1 hour."
                    )
                    parsed = None
                    break
                response_text = new_result["text"]
                usage_retry = new_result["usage"]

                # Accumulate usage again
                retry_ptc = usage_retry.prompt_token_count or 0
                retry_ctc = usage_retry.candidates_token_count or 0
                retry_ttc = usage_retry.total_token_count or 0
                total_prompt_tokens += retry_ptc
                total_candidates_tokens += retry_ctc
                total_tokens += retry_ttc

                logging.info(
                    f"[Retry usage] Additional tokens: input={retry_ptc}, "
                    f"candidate={retry_ctc}, total={retry_ttc} | "
                    f"New accumulated: input={total_prompt_tokens}, "
                    f"candidate={total_candidates_tokens}, total={total_tokens}"
                )
            else:
                break  # parse succeeded

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
    # Now gather *all* JSON files in the run_page_json_dir to create final CSV
    # -------------------------------------------------------------------------
    logging.info("Gathering all JSON files from page_by_page folder to build final CSV...")
    all_page_json_files = sorted(run_page_json_dir.glob("page_*.json"))

    merged_data: List[Any] = []

    for fpath in all_page_json_files:
        # page_0001.json => get '0001' => page_number=1
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

    # -------------------------------------------------------------------------
    # Add "id" to each record in merged_data (now truly all pages)
    # -------------------------------------------------------------------------
    for i, record in enumerate(merged_data, start=1):
        if isinstance(record, dict):
            record["id"] = i
        else:
            merged_data[i-1] = {"id": i, "value": str(record)}

    # -------------------------------------------------------------------------
    # Convert merged data (all pages) to CSV
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
            "candidates_tokens": total_candidates_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }

    with log_path.open("w", encoding="utf-8") as lf:
        json.dump(log_data, lf, indent=4)

    # -------------------------------------------------------------------------
    # Final token usage summary
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Total input tokens used: {total_prompt_tokens}")
    logging.info(f"Total candidate tokens used: {total_candidates_tokens}")
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