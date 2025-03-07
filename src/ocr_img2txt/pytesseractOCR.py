#!/usr/bin/env python3
"""
Pytesseract PDF -> PNG -> TEXT Pipeline

This script:
  1) Converts a PDF into per-page PNG images in data/page_by_page/PNG/<pdf_stem>.
     (Skips conversion if images already exist.)
  2) Performs OCR on each page using pytesseract (single attempt, no infinite retry).
  3) Merges all returned page texts into a single TXT file (<pdf_stem>.txt).
  4) Logs progress & timing information, and saves a JSON run log at the end.

"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from pdf2image import convert_from_path
from PIL import Image
import pytesseract

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "ocr_img2txt"
LOGS_DIR = PROJECT_ROOT / "logs" / "ocr_img2txt"

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
    Parse command-line arguments for the PDF-to-text pipeline using pytesseract.
    """
    parser = argparse.ArgumentParser(description="Pytesseract PDF-to-text pipeline")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF file in data/pdfs/, e.g. example.pdf"
    )
    return parser.parse_args()

###############################################################################
# Utility: Find existing run_XY directories to auto-increment run number
###############################################################################
def find_existing_runs_in_folder(model_folder: Path) -> List[int]:
    """
    Look for existing 'run_XX' directories in the given folder.
    Returns a list of run numbers (integers).
    """
    if not model_folder.is_dir():
        return []
    runs = []
    for child in model_folder.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                run_num = int(child.name.split("_")[1])
                runs.append(run_num)
            except ValueError:
                pass
    return runs

###############################################################################
# Utility: Write a JSON log file in logs/ocr_img2txt/<model_name>/
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file with run metadata in logs/ocr_img2txt/<model_name>/.
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
    Main function for PDF-to-text pipeline using pytesseract:

    Steps:
      1. Parse arguments & configure logging.
      2. Convert PDF => PNG in data/page_by_page/PNG/<pdf_stem>.
      3. Create results folder under results/ocr_img2txt/pytesseract/<pdf_stem>/run_xy/page_by_page/.
      4. Perform OCR (single attempt) on each page, saving <page_id>.txt files.
      5. Concatenate all page text files into <pdf_stem>.txt.
      6. Write a JSON log in logs/ocr_img2txt/pytesseract/.
    """
    # -------------------------------------------------------------------------
    # 1. Parse arguments and configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf
    model_name = "pytesseract"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== Pytesseract PDF -> PNG -> TEXT Pipeline ===")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model: {model_name}")

    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # 2. Convert PDF to page images in data/page_by_page/PNG/<pdf_stem>
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = PROJECT_ROOT / "data" / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"Could not find PDF file: {pdf_path}")
        sys.exit(1)

    png_dir = PROJECT_ROOT / "data" / "page_by_page" / "PNG" / pdf_stem
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

    png_files = sorted(png_dir.glob("page_*.png"))
    if not png_files:
        logging.error(f"No page images found in {png_dir}. Exiting.")
        sys.exit(1)

    total_pages = len(png_files)

    # -------------------------------------------------------------------------
    # 3. Create the results folder:
    #    results/ocr_img2txt/pytesseract/<pdf_stem>/run_xy/page_by_page/
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / model_name / pdf_stem
    base_results_path.mkdir(parents=True, exist_ok=True)

    existing_runs = find_existing_runs_in_folder(base_results_path)
    next_run_number = (max(existing_runs) + 1) if existing_runs else 1

    run_dir = base_results_path / f"run_{str(next_run_number).zfill(2)}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # 4. For each page, run pytesseract OCR (single attempt, no infinite retry)
    # -------------------------------------------------------------------------
    page_text_files = []
    for idx, png_path in enumerate(png_files, start=1):
        logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

        # Log DPI & size
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

                # Single attempt at OCR using deu_frak
                try:
                    transcription = pytesseract.image_to_string(pil_image, lang="deu_frak")
                except Exception as e:
                    logging.error(f"pytesseract OCR error on {png_path.stem}: {e}. Skipping this page.")
                    transcription = ""

        except Exception as e:
            logging.error(f"Could not open {png_path}: {e}. Skipping this page.")
            transcription = ""

        # If transcription is empty or None, log a warning
        if not transcription:
            logging.warning(f"Received empty OCR result for {png_path.stem}. Saving empty file.")

        # Save page_xxxx.txt
        out_txt_path = run_page_dir / f"{png_path.stem}.txt"
        out_txt_path.write_text(transcription, encoding="utf-8")
        page_text_files.append(out_txt_path)

        # ---------------------------------------------------------------------
        # Timing / estimation
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

    logging.info("All pages processed. Individual text files created.")

    # -------------------------------------------------------------------------
    # 5. Concatenate all page text files into <pdf_stem>.txt in the run folder
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            text_content = txt_file.read_text(encoding='utf-8').strip()
            outf.write(text_content + "\n\n")

    logging.info(f"Final concatenated file: {final_txt_path}")

    # -------------------------------------------------------------------------
    # 6. Write a JSON log summarizing the run
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start_time
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": model_name,
        "run_directory": str(run_dir),
        "pages_count": len(page_text_files),
        "final_text_file": str(final_txt_path),
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration)
    }

    write_json_log(log_info, model_name)
    logging.info("Run information successfully logged.")

    # Final summary
    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")


###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    main()