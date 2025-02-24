#!/usr/bin/env python3
"""
Gemini-2.0 PDF -> PNG -> TEXT Pipeline

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls Gemini-2.0 for each page image, producing text output.
     - Automatically retries on any error, up to 1 hour per page. If it fails after 1 hour, skips that page.
  3) Merges all returned page texts into a single TXT file (<pdf_stem>.txt).
  4) Logs usage tokens (prompt/candidate) per page, accumulates them across all pages, and saves a JSON run log.

Everything is aligned with your other Gemini-2.0 scripts for consistency.
"""

import argparse
import json
import logging
import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_img2txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_img2txt"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_img2txt"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load environment variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")  # Must match your .env key

###############################################################################
# Constants
###############################################################################
MODEL_NAME = "gemini-2.0"
FULL_MODEL_NAME = "gemini-2.0-flash" #    "gemini-2.0-flash-exp"
MAX_OUTPUT_TOKENS = 8192
RETRY_LIMIT_SECONDS = 3600  # 1 hour per page

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
    Parse command-line arguments for the Gemini-2.0 PDF-to-text pipeline.
    """
    parser = argparse.ArgumentParser(description="Gemini-2.0 PDF-to-text pipeline")

    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF file in data/pdfs/, e.g. example.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the LLM call (default: 0.0)"
    )

    return parser.parse_args()

###############################################################################
# Utility: Find existing run_XY directories to auto-increment run number
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
# Utility: Write a JSON log file in logs/llm_img2txt/gemini-2.0/
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
# Main Pipeline
###############################################################################
def main() -> None:
    """
    Main function for Gemini-2.0 PDF-to-text pipeline, with:
      - PNG conversion in data/page_by_page/PNG/<pdf_stem>
      - 1-hour max retry per page
      - Usage token logging
      - Final text concatenation
      - JSON run log with total usage stats
    """
    # -------------------------------------------------------------------------
    # 1. Parse arguments and configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf
    temperature = args.temperature

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== Gemini-2.0 PDF -> PNG -> TEXT Pipeline ===")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model: {MODEL_NAME}, Full model: {FULL_MODEL_NAME}")
    logging.info(f"Temperature: {temperature}")

    # Start overall timer
    overall_start = time.time()

    # -------------------------------------------------------------------------
    # 2. Load the transcription prompt from prompts/llm_img2txt/gemini-2.0.txt
    # -------------------------------------------------------------------------
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)

    transcription_prompt = prompt_path.read_text(encoding='utf-8').strip()
    if not transcription_prompt:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)
    logging.info(f"Prompt loaded from: {prompt_path}")

    # -------------------------------------------------------------------------
    # 3. Convert PDF to page images in data/page_by_page/PNG/<pdf_stem>
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    png_dir = DATA_DIR / "page_by_page" / "PNG" / pdf_stem
    if not png_dir.is_dir():
        logging.info(f"No PNG folder found; converting PDF -> PNG in {png_dir} ...")
        png_dir.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(pdf_path))
        for i, page_img in enumerate(pages, start=1):
            img_path = png_dir / f"page_{i:04d}.png"
            page_img.save(img_path, "PNG")
        logging.info(f"Created {len(pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"Folder {png_dir} already exists; skipping PDF->PNG step.")

    # Gather PNG paths
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error(f"No page images found in {png_dir}. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)

    # -------------------------------------------------------------------------
    # 4. Create results folder => results/llm_img2txt/gemini-2.0/<pdf_stem>/temperature_x.x/run_nn/page_by_page
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / MODEL_NAME / pdf_stem
    temp_dir = base_results_path / f"temperature_{temperature}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = find_existing_runs_in_temperature_folder(temp_dir)
    next_run_number = (max(existing_runs) + 1) if existing_runs else 1
    run_dir = temp_dir / f"run_{str(next_run_number).zfill(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # 5. For each page, call Gemini-2.0 with up to 1-hour retry + usage logging
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0

    page_text_files = []

    for idx, png_path in enumerate(png_files, start=1):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        try:
            with Image.open(png_path) as pil_image:
                # Log DPI (if known)
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

                # Attempt to transcribe with a 1-hour max retry
                start_retry = time.time()
                transcription = None
                page_usage_prompt = 0
                page_usage_candidate = 0
                page_usage_total = 0

                while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
                    tmp_file = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp_file = tmp.name
                            pil_image.save(tmp_file, "PNG")

                        client = genai.Client(api_key=API_KEY)
                        file_upload = client.files.upload(path=tmp_file)

                        response = client.models.generate_content(
                            model=FULL_MODEL_NAME,
                            contents=[
                                types.Part.from_uri(
                                    file_uri=file_upload.uri,
                                    mime_type=file_upload.mime_type,
                                ),
                                transcription_prompt
                            ],
                            config=types.GenerateContentConfig(
                                temperature=temperature,
                                max_output_tokens=MAX_OUTPUT_TOKENS,
                            ),
                        )

                        if not response or not response.text:
                            logging.warning(
                                "Gemini-2.0 returned empty response; retrying..."
                            )
                            continue

                        transcription = response.text
                        usage = response.usage_metadata

                        # Update usage
                        page_usage_prompt = usage.prompt_token_count or 0
                        page_usage_candidate = usage.candidates_token_count or 0
                        page_usage_total = usage.total_token_count or (
                            page_usage_prompt + page_usage_candidate
                        )

                        total_prompt_tokens += page_usage_prompt
                        total_candidates_tokens += page_usage_candidate
                        total_tokens += page_usage_total

                    except Exception as e:
                        logging.warning(f"Gemini-2.0 call failed: {e}. Retrying...")
                        continue
                    finally:
                        if tmp_file and os.path.exists(tmp_file):
                            try:
                                os.remove(tmp_file)
                            except:
                                pass

                    # If we made it here, we have a valid transcription
                    break

                else:
                    # If we exit the while-loop normally, 1 hour was exceeded
                    logging.error(
                        f"Skipping page {idx} because Gemini-2.0 API did not succeed within 1 hour."
                    )
                    logging.info("")
                    continue

        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            logging.info("")
            continue

        # If transcription is empty
        if not transcription:
            transcription = ""
            logging.warning(f"Received empty/None transcription for page {idx}.")

        # Log usage for this page
        logging.info(
            f"Gemini-2.0 usage for page {idx}: "
            f"input={page_usage_prompt}, candidate={page_usage_candidate}, total={page_usage_total}"
        )
        logging.info(
            f"Accumulated so far: input={total_prompt_tokens}, "
            f"candidate={total_candidates_tokens}, total={total_tokens}"
        )

        # Save page text
        page_text_path = run_page_dir / f"{png_path.stem}.txt"
        with open(page_text_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        page_text_files.append(page_text_path)

        # Timing / estimation
        elapsed = time.time() - overall_start
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

    logging.info("All pages processed. Individual text files created.")

    # -------------------------------------------------------------------------
    # 6. Concatenate all page text files into <pdf_stem>.txt in the run folder
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")

    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for text_file in sorted(page_text_files):
            outf.write(Path(text_file).read_text(encoding='utf-8').strip())
            outf.write("\n\n")  # separate pages with a blank line

    logging.info(f"Final concatenated file: {final_txt_path}")

    # -------------------------------------------------------------------------
    # 7. Write a JSON log summarizing the run
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature": temperature,
        "run_directory": str(run_dir),
        "prompt_file": str(prompt_path),
        "pages_count": len(page_text_files),
        "final_text_file": str(final_txt_path),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidates_tokens": total_candidates_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }
    write_json_log(log_info, MODEL_NAME)

    # Final usage summary
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Total input (prompt) tokens: {total_prompt_tokens}")
    logging.info(f"Total candidate tokens: {total_candidates_tokens}")
    logging.info(f"Grand total tokens: {total_tokens}")

    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()