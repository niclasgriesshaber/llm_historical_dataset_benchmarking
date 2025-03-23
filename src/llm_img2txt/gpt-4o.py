#!/usr/bin/env python3
"""
GPT-4o PDF -> PNG -> TEXT Pipeline

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Calls GPT-4o for each page image, retrieving text output.
     - Automatically retries on any error, up to 1 hour per page; if still failing, it skips that page.
  3) Merges all returned page texts into a single TXT file (<pdf_stem>.txt).
  4) Logs usage tokens per page, accumulates them across all pages, and saves a JSON run log.

All references are to GPT-4o, and the pipeline structure matches your other scripts.
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

import requests
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
API_KEY = os.getenv("OPENAI_API_KEY")  # Must match your .env key

###############################################################################
# Model Constants
###############################################################################
MODEL_NAME = "gpt-4o"
FULL_MODEL_NAME = "gpt-4o-2024-05-13"
MAX_OUTPUT_TOKENS = 4096
SEED = 42  # Not used by OpenAI, but included for consistency
RETRY_LIMIT_SECONDS = 3600  # 1 hour per page

###############################################################################
# Utility: Time Formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a number of seconds into H:MM:SS for consistent logging.
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
    Parse command-line arguments for a GPT-4o PDF-to-text pipeline.
    """
    parser = argparse.ArgumentParser(description="GPT-4o PDF-to-text pipeline")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. example.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the LLM call (default 0.0)"
    )
    return parser.parse_args()

###############################################################################
# Utility: Find existing run_XY directories
###############################################################################
def find_existing_runs_in_temperature_folder(temp_folder: Path) -> List[int]:
    """
    Scan for run_XX subfolders in temp_folder, returning a list of integers.
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
# Utility: Write a JSON log
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in logs/llm_img2txt/<model_name>/.
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
# OpenAI GPT-4o Call
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
    Call OpenAI's GPT-4o with an image + text prompt, returning (text_out, usage_dict).

    usage_dict has e.g. {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}.
    If there's an error, return (None, {}).
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
        "model": full_model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, json=payload)
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
# Main
###############################################################################
def main() -> None:
    """
    Main GPT-4o PDF -> PNG -> TEXT pipeline with:
      - data/page_by_page/PNG/<pdf_stem> for images
      - 1-hour max retry per page
      - usage token logging (input, candidate, total)
      - final text concatenation
      - JSON log & final usage summary
    """
    # -------------------------------------------------------------------------
    # Parse arguments, configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf
    temperature = args.temperature

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== GPT-4o PDF -> PNG -> TEXT Pipeline ===")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model: {MODEL_NAME}, Full model: {FULL_MODEL_NAME}")
    logging.info(f"Temperature: {temperature} | Seed={SEED}")

    # Overall timing start
    overall_start = time.time()

    # -------------------------------------------------------------------------
    # Load transcription prompt from: prompts/llm_img2txt/gpt-4o.txt
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
    # Convert PDF -> PNG in data/page_by_page/PNG/<pdf_stem>
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
            out_png = png_dir / f"page_{i:04d}.png"
            page_img.save(out_png, "PNG")
        logging.info(f"Created {len(pages)} PNG pages in {png_dir}")
    else:
        logging.info(f"Folder {png_dir} already exists; skipping PDF->PNG step.")

    # Gather all PNGs
    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error(f"No PNG pages found in {png_dir}. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)

    # -------------------------------------------------------------------------
    # Prepare results folder => results/llm_img2txt/gpt-4o/<pdf_stem>/temperature_x.x/run_nn/page_by_page
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / MODEL_NAME / pdf_stem
    temp_folder = base_results_path / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    existing_runs = find_existing_runs_in_temperature_folder(temp_folder)
    next_run = max(existing_runs) + 1 if existing_runs else 1
    run_dir = temp_folder / f"run_{str(next_run).zfill(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # Accumulate usage
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_completion_tokens = 0  # "candidate" tokens in your other scripts
    total_tokens = 0

    page_text_files = []

    # -------------------------------------------------------------------------
    # Process each PNG
    # -------------------------------------------------------------------------
    for idx, png_path in enumerate(png_files, start=1):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        # Open the image, log metadata
        try:
            with Image.open(png_path) as pil_image:
                width, height = pil_image.size
                dpi_value = pil_image.info.get("dpi", None)
                if dpi_value and len(dpi_value) == 2:
                    logging.info(f"Image metadata -> width={width}px, height={height}px, dpi={dpi_value}")
                else:
                    logging.info(f"Image metadata -> width={width}px, height={height}px, dpi=UNKNOWN")

                # 1-hour retry
                start_retry = time.time()
                text_out = None
                page_prompt = 0
                page_candidate = 0
                page_total = 0

                while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
                    try:
                        returned_text, usage_dict = openai_api(
                            prompt=transcription_prompt,
                            pil_image=pil_image,
                            full_model_name=FULL_MODEL_NAME,
                            max_tokens=MAX_OUTPUT_TOKENS,
                            temperature=temperature,
                            api_key=API_KEY
                        )

                        if not returned_text:
                            logging.warning("GPT-4o returned None or empty text; retrying...")
                            continue

                        text_out = returned_text
                        page_prompt = usage_dict.get("prompt_tokens", 0)
                        page_candidate = usage_dict.get("completion_tokens", 0)
                        page_total = usage_dict.get("total_tokens", page_prompt + page_candidate)

                        # Update accumulators
                        total_prompt_tokens += page_prompt
                        total_completion_tokens += page_candidate
                        total_tokens += page_total

                    except Exception as e:
                        logging.warning(f"GPT-4o call failed: {e}. Retrying...")
                        continue
                    else:
                        # success => break
                        break
                else:
                    # 1 hour was exceeded
                    logging.error(
                        f"Skipping page {idx} because GPT-4o call did not succeed within 1 hour."
                    )
                    logging.info("")
                    continue

        except Exception as e:
            logging.error(f"Failed to open image {png_path}: {e}")
            logging.info("")
            continue

        # If we never got any text
        if not text_out:
            text_out = ""
            logging.warning(f"Empty response for page {idx}. Saving empty file.")

        # Log usage for this page
        logging.info(
            f"GPT-4o usage for page {idx}: "
            f"input={page_prompt}, candidate={page_candidate}, total={page_total}"
        )
        logging.info(
            f"Accumulated so far: input={total_prompt_tokens}, "
            f"candidate={total_completion_tokens}, total={total_tokens}"
        )

        # Save page-level text
        page_txt_path = run_page_dir / f"{png_path.stem}.txt"
        page_txt_path.write_text(text_out, encoding='utf-8')
        page_text_files.append(page_txt_path)

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
    # Concatenate page texts => <pdf_stem>.txt in run_dir
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            content = txt_file.read_text(encoding='utf-8').strip()
            outf.write(content + "\n\n")

    logging.info(f"Final concatenated file: {final_txt_path}")

    # -------------------------------------------------------------------------
    # Write JSON log
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature": temperature,
        "seed": SEED,
        "run_directory": str(run_dir),
        "prompt_file": str(prompt_path),
        "pages_count": len(page_text_files),
        "final_text_file": str(final_txt_path),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidate_tokens": total_completion_tokens,
            "total_tokens": total_tokens
        },
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }
    write_json_log(log_info, MODEL_NAME)

    # Final usage summary
    logging.info("=== Final Usage Summary ===")
    logging.info(f"Total input (prompt) tokens used: {total_prompt_tokens}")
    logging.info(f"Total candidate tokens used: {total_completion_tokens}")
    logging.info(f"Grand total of all tokens used: {total_tokens}")

    # Final log
    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


if __name__ == "__main__":
    main()