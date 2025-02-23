#!/usr/bin/env python3
"""
o1 Post-Correction Pipeline (reading existing Transkribus OCR text).

This script:
  1) Ensures each page of the PDF has a PNG image in data/page_by_page/PNG/<pdf_stem>.
     - If not found, converts the PDF into per-page PNG images.
  2) For each page:
     - Loads existing OCR text from:
       results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
     - Calls the o1 reasoning model (with up to 1-hour retry) passing:
       - The page image (PNG).
       - The existing OCR text, as context in a "post-correction" prompt.
     - Stores the output as page_000N.txt in a new run folder.
  3) Merges all per-page outputs into a single TXT file (<pdf_stem>.txt).
  4) Logs usage tokens (prompt/completion) per page, accumulates them, and saves a JSON run log.
  5) If o1 fails on any page, the script stops immediately.

We keep "temperature_0.0" in the folder structure for compatibility with existing evals,
but do NOT pass any temperature parameter to the model.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_img2txt"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_img2txt"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

# We store PNG in data/page_by_page/PNG/<pdf_stem>
# Transkribus OCR text is pre-existing in:
# results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt

###############################################################################
# Load environment variables (for the o1 API key)
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("OPENAI_API_KEY")  # Must match your .env key

###############################################################################
# Model Config for o1
###############################################################################
MODEL_NAME = "o1-with-transkribus"  # Folder name for final outputs
FULL_MODEL_NAME = "o1"              # Actual model name passed to the API
RETRY_LIMIT_SECONDS = 3600          # 1 hour max retry per page

# We store "temperature_0.0" in the folder name for compatibility, but do not use it in the API call
FOLDER_TEMPERATURE = "0.0"

# Reasoning parameters:
REASONING_CONFIG = {
    "reasoning_effort": "high",      # "low", "medium", or "high"
    "max_completion_tokens": 100000  # Large limit to allow for reasoning + final output
}

SEED = 42  # Not used by o1, but kept for reference

###############################################################################
# Utility: Time Formatting
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
# Argument Parsing
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments: only --pdf. We do NOT parse temperature.
    """
    parser = argparse.ArgumentParser(
        description="o1 post-correction pipeline (using existing Transkribus OCR)."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF file in data/pdfs/, e.g. example.pdf"
    )
    return parser.parse_args()

###############################################################################
# Utility: Find existing 'run_XX' directories to auto-increment
###############################################################################
def find_existing_runs_in_temperature_folder(temp_folder: Path) -> List[int]:
    """
    Look for existing 'run_XX' directories in the temperature-specific folder.
    Returns a list of run numbers (integers).
    """
    if not temp_folder.is_dir():
        return []
    runs = []
    for child in temp_folder.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                run_num = int(child.name.split("_")[1])
                runs.append(run_num)
            except ValueError:
                pass
    return runs

###############################################################################
# Utility: Write a JSON log file in logs/llm_img2txt/<model_name>/
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in the logs/llm_img2txt/<model_name>/ folder.
    """
    pipeline_logs_dir = LOGS_DIR / model_name
    pipeline_logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp_str}.json"
    log_path = pipeline_logs_dir / log_filename

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_dict, f, indent=4)

    logging.info(f"JSON log saved at: {log_path}")

###############################################################################
# Load a base prompt (if needed)
###############################################################################
def load_base_prompt() -> str:
    """
    Reads the prompt text from:
    src/prompts/llm_img2txt/o1.txt
    or any other name you wish to use,
    and returns it as a string.
    """
    prompt_path = PROJECT_ROOT / "src" / "prompts" / "llm_img2txt" / "o1.txt"
    if not prompt_path.is_file():
        logging.error(f"Prompt file not found at {prompt_path}")
        sys.exit(1)
    return prompt_path.read_text(encoding='utf-8')

###############################################################################
# Multimodal OCR correction prompt construction
###############################################################################
def build_correction_prompt(ocr_text: str, image_path: Path) -> str:
    """
    Applies multimodal OCR post-correction logic while enforcing strict transcription rules.

    :param ocr_text: Raw text extracted from OCR (Transkribus output)
    :param image_path: Path to the corresponding image for reference
    :return: Full prompt to pass to the model
    """
    # Load the base prompt from the text file
    base_prompt = load_base_prompt().strip()

    # Merge the base prompt and the OCR text
    # Adjust or rename as you see fit:
    combined_prompt = (
        f"{base_prompt}\n\n"
        "Below is the OCR output from Transkribus. Please correct any errors, ensuring the text is fully "
        "compliant with the strict transcription rules.\n\n"
        "-- OCR Output (Transkribus) --\n"
        f"{ocr_text}\n"
    )

    return combined_prompt

###############################################################################
# o1 API Call
###############################################################################
def openai_o1_api(
    prompt: str,
    pil_image: Image.Image,
    api_key: str
) -> (Optional[str], dict):
    """
    Call OpenAI's o1 model with an image + text prompt, returning (text_out, usage_dict).

    usage_dict may have keys like:
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

    # Convert image to base64 PNG
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    with BytesIO() as buffer:
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": FULL_MODEL_NAME,  # "o1"
        "reasoning_effort": REASONING_CONFIG["reasoning_effort"],
        "max_completion_tokens": REASONING_CONFIG["max_completion_tokens"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
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
# Main Pipeline
###############################################################################
def main() -> None:
    """
    1) Convert PDF => PNG (if not already present) in data/page_by_page/PNG/<pdf_stem>.
    2) For each page:
       - Load pre-existing OCR text from results/ocr_img2txt/transkribus/<pdf_stem>/run_01/page_by_page/page_000N.txt
       - Call o1 => post-correct using the OCR text + image (1-hour max retry).
       - Save result in a new run folder.
    3) Write per-page results and a merged final text file.
    4) Write JSON usage log.
    5) Stop if any step fails on any page.
    """
    # -------------------------------------------------------------------------
    # 1. Parse arguments & configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== o1 Post-Correction Pipeline ===")
    logging.info(f"PDF: {pdf_name}")
    logging.info(f"Folder structure is set to temperature_{FOLDER_TEMPERATURE} (not used by model).")
    logging.info(f"Seed (unused by model, for reference): {SEED}")

    # Check environment for the required API key
    if not API_KEY:
        logging.error("OPENAI_API_KEY not set in .env or environment.")
        sys.exit(1)

    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # 2. Prepare PDF path & create PNG if needed
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at {pdf_path}")
        sys.exit(1)

    # PNG folder
    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"Converting PDF -> PNG in {png_dir}")
        png_dir.mkdir(parents=True, exist_ok=True)

        png_pages = convert_from_path(str(pdf_path))
        for i, page_img in enumerate(png_pages, start=1):
            img_path = png_dir / f"page_{i:04d}.png"
            page_img.save(img_path, "PNG")
        logging.info(f"Created {len(png_pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"PNG folder already exists: {png_dir}")

    # Gather page list
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error("No PNG page images found. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)
    logging.info(f"Total pages to process: {total_pages}")

    # -------------------------------------------------------------------------
    # 3. Create results folder => results/llm_img2txt/o1-with-transkribus/<pdf_stem>/temperature_0.0/run_XX/page_by_page
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / MODEL_NAME / pdf_stem
    temp_dir = base_results_path / f"temperature_{FOLDER_TEMPERATURE}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = find_existing_runs_in_temperature_folder(temp_dir)
    next_run_number = (max(existing_runs) + 1) if existing_runs else 1

    run_dir = temp_dir / f"run_{str(next_run_number).zfill(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # 4. Initialize usage counters for o1
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    # If you want to track advanced details:
    total_reasoning_tokens = 0
    total_accepted_tokens = 0
    total_rejected_tokens = 0

    # -------------------------------------------------------------------------
    # 5. Main Loop: For each page, read existing OCR -> call o1
    # -------------------------------------------------------------------------
    page_text_files = []

    # Root folder for pre-existing Transkribus OCR text
    transkribus_ocr_root = (
        PROJECT_ROOT
        / "results"
        / "ocr_img2txt"
        / "transkribus"
        / pdf_stem
        / "run_01"
        / "page_by_page"
    )

    for idx, png_path in enumerate(png_files, start=1):
        page_num = idx
        logging.info(f"=== Page {page_num} of {total_pages} ===")
        logging.info(f"PNG: {png_path.name}")

        # 5a) Read the existing OCR text
        transkribus_txt_path = transkribus_ocr_root / f"page_{page_num:04d}.txt"
        if not transkribus_txt_path.is_file():
            logging.error(f"OCR text file not found: {transkribus_txt_path}")
            sys.exit(1)

        ocr_text = transkribus_txt_path.read_text(encoding='utf-8').strip()
        logging.info(f"Loaded OCR text (~{len(ocr_text)} chars).")

        # 5b) Construct the prompt using our function
        correction_prompt = build_correction_prompt(ocr_text, png_path)

        # We'll store the final text from the model
        corrected_text = None
        page_prompt_tokens = 0
        page_completion_tokens = 0
        page_total_tokens = 0
        page_reasoning_tokens = 0
        page_accepted_tokens = 0
        page_rejected_tokens = 0

        logging.info("Calling o1 for post-correction...")

        start_retry = time.time()
        while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
            try:
                with Image.open(png_path) as pil_img:
                    returned_text, usage_dict = openai_o1_api(
                        prompt=correction_prompt,
                        pil_image=pil_img,
                        api_key=API_KEY
                    )

                if not returned_text:
                    logging.warning("o1 returned empty response; retrying...")
                    continue

                corrected_text = returned_text

                # Extract usage
                page_prompt_tokens = usage_dict.get("prompt_tokens", 0)
                page_completion_tokens = usage_dict.get("completion_tokens", 0)
                page_total_tokens = usage_dict.get(
                    "total_tokens",
                    page_prompt_tokens + page_completion_tokens
                )

                ctd = usage_dict.get("completion_tokens_details", {})
                page_reasoning_tokens = ctd.get("reasoning_tokens", 0)
                page_accepted_tokens = ctd.get("accepted_prediction_tokens", 0)
                page_rejected_tokens = ctd.get("rejected_prediction_tokens", 0)

                # Update global usage
                total_prompt_tokens += page_prompt_tokens
                total_completion_tokens += page_completion_tokens
                total_tokens += page_total_tokens
                total_reasoning_tokens += page_reasoning_tokens
                total_accepted_tokens += page_accepted_tokens
                total_rejected_tokens += page_rejected_tokens

            except Exception as e:
                logging.warning(f"o1 call failed: {e}. Retrying...")
                continue

            # If we got valid output, break the retry loop
            if corrected_text:
                break
        else:
            # Exceeded 1 hour of retries => fail the pipeline
            logging.error(f"o1 call failed for page {page_num} after 1 hour. Stopping.")
            sys.exit(1)

        # 5c) Save the corrected output to page_X.txt
        page_text_path = run_page_dir / f"page_{page_num:04d}.txt"
        page_text_path.write_text(corrected_text, encoding='utf-8')
        page_text_files.append(page_text_path)

        # Log usage
        logging.info(
            f"o1 usage for page {page_num}: "
            f"prompt={page_prompt_tokens}, completion={page_completion_tokens}, total={page_total_tokens}, "
            f"reasoning={page_reasoning_tokens}, accepted_pred={page_accepted_tokens}, rejected_pred={page_rejected_tokens}"
        )
        logging.info(
            "Accumulated so far: "
            f"prompt={total_prompt_tokens}, completion={total_completion_tokens}, total={total_tokens}, "
            f"reasoning={total_reasoning_tokens}, accepted_pred={total_accepted_tokens}, "
            f"rejected_pred={total_rejected_tokens}"
        )

        # Timing / estimation
        elapsed = time.time() - overall_start_time
        pages_done = page_num
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

    logging.info("All pages processed successfully. Outputs saved.")

    # -------------------------------------------------------------------------
    # 6. Merge all page text files -> final <pdf_stem>.txt
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path}")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            text_content = txt_file.read_text(encoding='utf-8').strip()
            outf.write(text_content + "\n\n")

    # -------------------------------------------------------------------------
    # 7. Write JSON Log
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start_time
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "reasoning_config": REASONING_CONFIG,
        "folder_temperature": FOLDER_TEMPERATURE,
        "seed": SEED,
        "pages_count": total_pages,
        "final_text_file": str(final_txt_path),
        "run_directory": str(run_dir),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": total_reasoning_tokens,
            "accepted_prediction_tokens": total_accepted_tokens,
            "rejected_prediction_tokens": total_rejected_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }
    write_json_log(log_info, MODEL_NAME)

    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------
    logging.info("=== Final Usage Summary (including new reasoning fields) ===")
    logging.info(f"Prompt tokens:           {total_prompt_tokens}")
    logging.info(f"Completion tokens:       {total_completion_tokens}")
    logging.info(f"Reasoning tokens:        {total_reasoning_tokens}")
    logging.info(f"Accepted prediction:     {total_accepted_tokens}")
    logging.info(f"Rejected prediction:     {total_rejected_tokens}")
    logging.info(f"Grand total tokens used: {total_tokens}")
    logging.info(
        f"Completed in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()