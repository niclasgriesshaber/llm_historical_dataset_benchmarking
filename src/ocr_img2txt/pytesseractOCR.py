#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# IMPORTANT:
# If your script is named "pytesseract.py", it can overshadow the real pytesseract library.
# That can cause "module 'pytesseract' has no attribute 'image_to_string'" errors.
# One workaround is to rename this script to something else (e.g. "pytesseract_ocr.py").
# Another is to manipulate sys.path before importing. Shown here is the safer approach:
#   import as a different name and ensure we’re importing the actual package.

try:
    import pytesseract as real_tesseract
except ImportError as e:
    # If this fails, ensure that 'pytesseract' is installed and you haven't overshadowed it locally.
    print(f"Could not import pytesseract properly: {e}")
    sys.exit(1)

from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# Project Paths
###############################################################################
# In your structure, "pytesseract.py" is in: project_root/src/ocr_img2txt/pytesseract.py
# So .parents[2] should be the project root. Adjust if needed.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "ocr_img2txt"
LOGS_DIR = PROJECT_ROOT / "logs" / "ocr_img2txt"

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
        help="Name of the PDF file in data/pdfs/, e.g. type-1.pdf"
    )

    return parser.parse_args()


###############################################################################
# Utility: Find existing run_XY directories to auto-increment run number
###############################################################################
def find_existing_runs_in_folder(model_folder: Path) -> List[int]:
    """
    Look for existing 'run_XX' directories in the given folder.

    Args:
        model_folder (Path): The folder to scan.

    Returns:
        List[int]: A list of run numbers (integers) found in the folder.
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
# Utility: Write a JSON log file in logs/ocr_img2txt/
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in the logs directory.

    Args:
        log_dict (dict): The dictionary containing run metadata.
        model_name (str): The name of the OCR model used (e.g., 'pytesseract').
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
    Main function for PDF-to-text pipeline using pytesseract (no infinite retry).

    Steps:
      1. Parse arguments and configure logging.
      2. Convert PDF to page images and store them in data/page_by_page/<pdf_stem>/.
      3. Create results folder under:
         results/ocr_img2txt/pytesseract/<pdf_stem>/run_xy/page_by_page/
      4. For each page image, run pytesseract OCR (single attempt, no infinite retry),
         and save page_000X.txt.
      5. Concatenate all page text files into <pdf_stem>.txt.
      6. Write a JSON log in logs/ocr_img2txt/pytesseract/.
    """
    # -------------------------------------------------------------------------
    # 1. Parse arguments and configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()

    pdf_name = args.pdf            # e.g., "type-1.pdf"
    model_name = "pytesseract"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("Starting Pytesseract PDF-to-text pipeline...")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model name: {model_name}")

    # -------------------------------------------------------------------------
    # 2. Convert PDF to page images in data/page_by_page/<pdf_stem>/
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem  # e.g. "type-1"
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"Could not find PDF file: {pdf_path}")
        sys.exit(1)

    out_dir = DATA_DIR / "page_by_page" / pdf_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting PDF to images => {out_dir} ...")
    pages = convert_from_path(str(pdf_path))

    image_paths = []
    for i, page_img in enumerate(pages, start=1):
        img_name = f"page_{i:04}.png"
        img_path = out_dir / img_name
        page_img.save(img_path, "PNG")
        image_paths.append(img_path)

    logging.info(f"PDF split into {len(image_paths)} page images.")

    # -------------------------------------------------------------------------
    # 3. Create the results folder:
    #    results/ocr_img2txt/pytesseract/<pdf_stem>/run_xy/page_by_page/
    # -------------------------------------------------------------------------
    base_results_path = RESULTS_DIR / model_name / pdf_stem
    base_results_path.mkdir(parents=True, exist_ok=True)

    existing_runs = find_existing_runs_in_folder(base_results_path)
    next_run_number = (max(existing_runs) + 1) if existing_runs else 1

    run_dir_name = f"run_{str(next_run_number).zfill(2)}"
    run_dir = base_results_path / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=False)

    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # -------------------------------------------------------------------------
    # 4. For each page image, run pytesseract OCR (single attempt)
    # -------------------------------------------------------------------------
    page_text_files = []

    for img_path in image_paths:
        page_id = img_path.stem  # e.g. "page_0001"
        logging.info(f"Performing OCR on {page_id} ...")

        pil_image = Image.open(img_path)

        transcription = ""
        try:
            # Use the real pytesseract under the alias real_tesseract
            transcription = real_tesseract.image_to_string(pil_image)
        except Exception as e:
            # If an error occurs, log it and move on (no infinite retry)
            logging.error(f"pytesseract OCR error on {page_id}: {e}. Skipping this page.")
            transcription = ""

        # If transcription is empty or None, log a warning
        if not transcription:
            logging.warning(f"Received empty OCR result for {page_id}. Saving empty file.")

        # Save page_xxxx.txt
        out_txt_path = run_page_dir / f"{page_id}.txt"
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        page_text_files.append(out_txt_path)

    logging.info("All pages processed. Individual text files created.")

    # -------------------------------------------------------------------------
    # 5. Concatenate all page text files into <pdf_stem>.txt in the run folder
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            with open(txt_file, 'r', encoding='utf-8') as tf:
                outf.write(tf.read().strip())
                outf.write("\n\n")  # Separate pages by blank line

    logging.info(f"Final concatenated file: {final_txt_path}")

    # -------------------------------------------------------------------------
    # 6. Write a JSON log summarizing the run
    # -------------------------------------------------------------------------
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": model_name,
        "run_directory": str(run_dir),
        "pages_count": len(page_text_files),
        "final_text_file": str(final_txt_path),
    }

    write_json_log(log_info, model_name)
    logging.info("Run information successfully logged.")

    logging.info("Pipeline completed successfully. All done!")


###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    main()