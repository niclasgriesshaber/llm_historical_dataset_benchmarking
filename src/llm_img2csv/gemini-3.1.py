#!/usr/bin/env python3
"""
###############################################################################
# REVISION EXPERIMENT SCRIPT
#
# This script is an adaptation of the Gemini 2.0 img2csv pipeline for the
# Gemini 3.1 Pro Preview model. It is part of a revision experiment that
# stores results and logs in separate directories (results_revisions/ and
# logs_revisions/) to keep them isolated from the original experiment outputs.
#
# Changes from gemini-2.0.py:
#   - MODEL_NAME changed from "gemini-2.0" to "gemini-3.1"
#   - FULL_MODEL_NAME changed from "gemini-2.0-flash" to "gemini-3.1-pro-preview"
#   - RESULTS_DIR changed from "results/" to "results_revisions/"
#   - LOGS_DIR changed from "logs/" to "logs_revisions/"
#   - Prompt file changed from "gemini-2.0.txt" to "gemini-3.1.txt"
#
# All pipeline logic remains identical to the original.
###############################################################################

Gemini-3.1 PDF -> PNG -> JSON -> CSV Pipeline

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls Gemini-3.1 for each page image, retrieving JSON output.
     - If any API call or JSON parse fails, the script logs the error and exits immediately.
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

# Google-GenAI (Gemini) library -- used for both Gemini 2.0 and 3.1 models
import google.genai as genai
from google.genai import types

###############################################################################
# Project Paths
###############################################################################
# PROJECT_ROOT is two levels up from this script (src/llm_img2csv/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory where source PDFs are stored
DATA_DIR = PROJECT_ROOT / "data"

# Directory containing the prompt text files for the img2csv pipeline
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_img2csv"

# REVISION: Changed from "results" to "results_revisions" to isolate revision outputs
RESULTS_DIR = PROJECT_ROOT / "results_revisions" / "llm_img2csv"

# REVISION: Changed from "logs" to "logs_revisions" to isolate revision logs
LOGS_DIR = PROJECT_ROOT / "logs_revisions" / "llm_img2csv"

# Path to the .env file containing API keys
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
###############################################################################
# Load the .env file so we can access GOOGLE_API_KEY
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# REVISION: Changed from "gemini-2.0" to "gemini-3.1" -- used for folder naming and prompt file lookup
MODEL_NAME = "gemini-3.1"

# REVISION: Changed from "gemini-2.0-flash" to "gemini-3.1-pro-preview" -- the actual API model identifier
FULL_MODEL_NAME = "gemini-3.1-pro-preview"

# REVISION: Set to 65536 (maximum). Gemini 3.1 Pro is a reasoning model —
# internal thinking tokens count against the output budget, so we must
# set this to the maximum to avoid truncated output.
MAX_OUTPUT_TOKENS = 65536

###############################################################################
# Utility: Time formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a number of seconds into H:MM:SS for clean logging.

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
    Extract code-fenced JSON if present, otherwise fallback to raw text.
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
# Utility: Reorder dictionary keys with page_number at the end
###############################################################################
def reorder_dict_with_page_number(d: Dict[str, Any], page_number: int) -> Dict[str, Any]:
    """
    Return a new dict that includes a 'page_number' key at the end
    and ensures 'additional_information' is last if present.

    This standardizes the column order in the final CSV so that metadata
    columns (page_number, additional_information) appear after the data columns.

    Args:
        d: The original dictionary extracted from the model's JSON response.
        page_number: The page number to attach to this record.

    Returns:
        A new dictionary with keys reordered: data keys (sorted), then page_number,
        then additional_information (if present).
    """
    # Identify special keys that should be placed at the end
    special_keys = {"page_number", "additional_information"}
    # Collect all non-special keys and sort them alphabetically
    base_keys = [k for k in d.keys() if k not in special_keys]
    base_keys.sort()

    # Build the output dict with sorted base keys first
    out = {}
    for k in base_keys:
        out[k] = d[k]

    # Append page_number after the data keys
    out["page_number"] = page_number

    # If the model included an additional_information field, put it last
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
        # Fallback: treat as a single row with "value" column
        records = [{"value": str(json_data)}]

    # Collect all unique keys across all records to form CSV column headers
    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    # Place special columns at the end; sort all other columns alphabetically
    special_order = ["page_number", "additional_information"]
    base_keys = [k for k in all_keys if k not in special_order]
    base_keys.sort()
    fieldnames = base_keys + [k for k in special_order if k in all_keys]

    # Ensure the output directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the CSV file
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            if not isinstance(rec, dict):
                # Fallback: convert non-dict items to a row with empty fields + "value"
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
def gemini_api_call(
    prompt: str,
    pil_image: Image.Image,
    temperature: float
) -> dict:
    """
    Call Gemini 3.1 with the given prompt + image. If the call fails or
    returns empty results, log the error and exit immediately.

    The workflow:
      1) Save the PIL image to a temporary PNG file on disk.
      2) Upload that file to the GenAI file service.
      3) Send the uploaded file URI + prompt text to the model.
      4) If the model returns a valid text response, return it along with usage metadata.
      5) If anything fails, log the error and exit.

    Args:
        prompt: The text prompt to send to the model.
        pil_image: The page image as a PIL Image object.
        temperature: The sampling temperature for the model.

    Returns:
        A dict with "text" (the model's response) and "usage" (token usage metadata).
    """
    # Create a GenAI client using the API key from environment
    client = genai.Client(api_key=API_KEY)
    tmp_file = None
    try:
        # Step 1: Save the PIL image to a temporary file for upload
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_file = tmp.name
            pil_image.save(tmp_file, "PNG")

        # Step 2: Upload the image file to the GenAI file service
        file_upload = client.files.upload(file=tmp_file)

        # Step 3: Send the image + prompt to Gemini 3.1
        # REVISION: Uses FULL_MODEL_NAME "gemini-3.1-pro-preview" instead of "gemini-2.0-flash"
        response = client.models.generate_content(
            model=FULL_MODEL_NAME,
            contents=[
                # First content part: the uploaded image referenced by URI
                types.Part.from_uri(
                    file_uri=file_upload.uri,
                    mime_type=file_upload.mime_type,
                ),
                # Second content part: the text prompt
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                # Request JSON output format from the model
                response_mime_type="application/json",
                # REVISION: Dynamic thinking — lets the model decide how much
                # reasoning to use. Set to -1 for dynamic (comparable to
                # GPT-5.5's reasoning_effort="high").
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            ),
        )

        # Step 4: Validate the response
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
    finally:
        # Clean up the temporary file regardless of success or failure
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except:
                pass

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the Gemini-3.1 PDF-to-CSV pipeline.

    Pipeline steps:
      1) Parse command-line arguments (--pdf, --temperature, --continue_from_page)
      2) Configure logging to stdout
      3) Verify the specified PDF exists in data/pdfs/
      4) Load the task prompt from src/prompts/llm_img2csv/gemini-3.1.txt
      5) Convert the PDF to per-page PNG images (skip if already done)
      6) Create a run folder under results_revisions/llm_img2csv/gemini-3.1/...
      7) For each page: call Gemini 3.1 API, parse JSON response, save per-page JSON
      8) Merge all per-page JSON files into a single CSV
      9) Write a JSON run log to logs_revisions/llm_img2csv/gemini-3.1/
    """
    # -------------------------------------------------------------------------
    # Step 1: Parse command-line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Gemini-3.1 PDF-to-JSON-to-CSV Pipeline")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. my_file.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for Gemini-3.1 (default = 0.0)"
    )
    # --continue_from_page defaults to None so we can detect if it was provided
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
    # Step 2: Configure logging to stdout with timestamps
    # -------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # REVISION: Updated log header to reference Gemini-3.1
    logging.info("=== Gemini-3.1 PDF -> PNG -> JSON -> CSV Pipeline ===")
    logging.info(f"PDF: {pdf_name} | Temperature: {temperature}")

    # -------------------------------------------------------------------------
    # Step 3: Verify the PDF exists in data/pdfs/
    # -------------------------------------------------------------------------
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    # Extract the stem (filename without extension) for naming output files/folders
    pdf_stem = Path(pdf_name).stem

    # -------------------------------------------------------------------------
    # Step 4: Load the task prompt for Gemini-3.1
    # REVISION: Prompt file is now "gemini-3.1.txt" instead of "gemini-2.0.txt"
    # -------------------------------------------------------------------------
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Missing prompt file: {prompt_path}")
        sys.exit(1)

    # Read the entire prompt file as a string
    task_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not task_prompt:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)

    logging.info(f"Loaded Gemini-3.1 prompt from: {prompt_path}")

    # -------------------------------------------------------------------------
    # Step 5: Convert PDF to per-page PNG images (skip if directory already exists)
    # Images are stored in data/page_by_page/PNG/<pdf_stem>/
    # -------------------------------------------------------------------------
    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"No PNG folder found; converting PDF -> PNG in {png_dir} ...")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Use pdf2image to convert each page of the PDF to a PIL Image
        pages = convert_from_path(str(pdf_path))
        for i, pil_img in enumerate(pages, start=1):
            out_png = png_dir / f"page_{i:04d}.png"
            pil_img.save(out_png, "PNG")
        logging.info(f"Created {len(pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"Folder {png_dir} already exists; skipping PDF->PNG step.")

    # Gather all PNG files sorted by name (page_0001.png, page_0002.png, ...)
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error(f"No PNG pages found in {png_dir}. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)

    # -------------------------------------------------------------------------
    # Step 6: Prepare results folder structure
    # REVISION: Results go to results_revisions/ instead of results/
    # Path: results_revisions/llm_img2csv/gemini-3.1/<pdf_stem>/temperature_x.x/run_nn/
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

    # -------------------------------------------------------------------------
    # Run folder logic: either create a new run or continue an existing one
    # -------------------------------------------------------------------------
    if continue_from_page is None:
        # No --continue_from_page given: always start a fresh run
        next_run = highest_run_num + 1
        run_folder = temp_folder / f"run_{str(next_run).zfill(2)}"
        run_folder.mkdir(parents=True, exist_ok=False)
        logging.info(f"No --continue_from_page given; created new run folder: {run_folder}")
    else:
        # --continue_from_page was provided: reuse the highest existing run folder
        if highest_run_num == 0:
            # No runs exist yet, so create run_01
            run_folder = temp_folder / "run_01"
            run_folder.mkdir(parents=True, exist_ok=True)
            logging.info(
                f"No existing runs found, but --continue_from_page={continue_from_page} set. Using {run_folder}."
            )
        else:
            # Continue with the most recent run folder
            run_folder = temp_folder / f"run_{str(highest_run_num).zfill(2)}"
            logging.info(
                f"Continuing from page {continue_from_page}, using existing run folder: {run_folder}"
            )

    # Subfolder for per-page JSON output files
    run_page_json_dir = run_folder / "page_by_page"
    run_page_json_dir.mkdir(parents=True, exist_ok=True)

    # Final merged CSV path within the run folder
    final_csv_path = run_folder / f"{pdf_stem}.csv"

    # -------------------------------------------------------------------------
    # Initialize accumulators for token usage tracking
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0

    # -------------------------------------------------------------------------
    # Start timing the entire pipeline
    # -------------------------------------------------------------------------
    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 7: Process each page image through Gemini 3.1
    # For each page: call the API, parse the JSON, save per-page JSON
    # -------------------------------------------------------------------------
    # Determine the starting page (either page 1 or the specified continue page)
    start_page = continue_from_page if continue_from_page is not None else 1
    if start_page > 1:
        # Skip pages before the start page by slicing the file list
        png_files = png_files[start_page - 1:]

    for idx, png_path in enumerate(png_files, start=start_page):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        # Log image metadata (dimensions and DPI) before calling the API
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

                # Call Gemini 3.1 API (no retry -- exits on failure)
                result = gemini_api_call(
                    prompt=task_prompt,
                    pil_image=pil_image,
                    temperature=temperature
                )
        except SystemExit:
            raise
        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            sys.exit(1)

        # Extract the text response and usage metadata from the API result
        response_text = result["text"]
        usage_meta = result["usage"]

        # ---------------------------------------------------------------------
        # Update token usage accumulators for this page
        # ---------------------------------------------------------------------
        page_prompt_tokens = usage_meta.prompt_token_count or 0
        page_candidate_tokens = usage_meta.candidates_token_count or 0
        page_total_tokens = usage_meta.total_token_count or 0

        total_prompt_tokens += page_prompt_tokens
        total_candidates_tokens += page_candidate_tokens
        total_tokens += page_total_tokens

        logging.info(
            f"Gemini-3.1 usage for page {idx}: "
            f"input={page_prompt_tokens}, candidate={page_candidate_tokens}, total={page_total_tokens}"
        )
        logging.info(
            f"Accumulated usage so far: input={total_prompt_tokens}, "
            f"candidate={total_candidates_tokens}, total={total_tokens}"
        )

        # ---------------------------------------------------------------------
        # Parse the JSON from the model's text response (no retry)
        # If parsing fails, log the error and exit immediately
        # ---------------------------------------------------------------------
        try:
            parsed = parse_json_str(response_text)
        except ValueError as ve:
            logging.error(f"JSON parse error for page {idx}: {ve}")
            logging.error(f"Response text: \n{response_text}\n")
            sys.exit(1)

        # ---------------------------------------------------------------------
        # Save the parsed JSON for this page to a per-page file
        # File name matches the PNG name: e.g., page_0001.json
        # ---------------------------------------------------------------------
        page_json_path = run_page_json_dir / f"{png_path.stem}.json"
        with page_json_path.open("w", encoding="utf-8") as jf:
            json.dump(parsed, jf, indent=2, ensure_ascii=False)

        # ---------------------------------------------------------------------
        # Log timing and estimated time remaining
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
    # Step 8: Merge all per-page JSON files into a single CSV
    # This reads ALL JSON files in the page_by_page directory (including any
    # from a previous partial run if --continue_from_page was used)
    # -------------------------------------------------------------------------
    logging.info("Gathering all JSON files from page_by_page folder to build final CSV...")
    all_page_json_files = sorted(run_page_json_dir.glob("page_*.json"))

    merged_data: List[Any] = []

    for fpath in all_page_json_files:
        # Extract page number from filename: page_0001.json -> page_number=1
        page_str = fpath.stem.split("_")[1]
        page_num = int(page_str)

        # Load the per-page JSON data
        with fpath.open("r", encoding="utf-8") as jf:
            content = json.load(jf)

        # Normalize the content: could be a list of records or a single record
        if isinstance(content, list):
            for obj in content:
                if isinstance(obj, dict):
                    # Reorder keys so page_number appears at the end
                    merged_data.append(reorder_dict_with_page_number(obj, page_num))
                else:
                    merged_data.append(obj)
        elif isinstance(content, dict):
            merged_data.append(reorder_dict_with_page_number(content, page_num))
        else:
            merged_data.append(content)

    # -------------------------------------------------------------------------
    # Add sequential "id" to each record for the final CSV
    # -------------------------------------------------------------------------
    for i, record in enumerate(merged_data, start=1):
        if isinstance(record, dict):
            record["id"] = i
        else:
            # Wrap non-dict items in a dict with id and value
            merged_data[i-1] = {"id": i, "value": str(record)}

    # -------------------------------------------------------------------------
    # Write the merged data to a single CSV file
    # -------------------------------------------------------------------------
    convert_json_to_csv(merged_data, final_csv_path)
    logging.info(f"Final CSV (all pages) saved at: {final_csv_path}")

    # -------------------------------------------------------------------------
    # Step 9: Write JSON log with run metadata
    # REVISION: Logs go to logs_revisions/ instead of logs/
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
