#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from typing import List

# For PDF -> image conversion
from pdf2image import convert_from_path

# Transkribus
from transkribus_metagrapho_api import transkribus_metagrapho_api

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "ocr_img2txt"
LOGS_DIR = PROJECT_ROOT / "logs" / "ocr_img2txt"

# .env file is in config/.env
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Transkribus environment variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)

TRANSKRIBUS_USERNAME = os.getenv("TRANSKRIBUS_USERNAME")
TRANSKRIBUS_PASSWORD = os.getenv("TRANSKRIBUS_PASSWORD")

# --------------------------------------------------------------------
# Convert these IDs to int so Transkribus API recognizes them properly
# --------------------------------------------------------------------
line_det_str = os.getenv("TRANSKRIBUS_LINE_DETECTION_ID")
htr_id_str = os.getenv("TRANSKRIBUS_HTR_ID")

if not line_det_str or not htr_id_str:
    logging.error("Missing TRANSKRIBUS_LINE_DETECTION_ID or TRANSKRIBUS_HTR_ID in .env.")
    sys.exit(1)

try:
    TRANSKRIBUS_LINE_DETECTION_ID = int(line_det_str)  # <-- FIX HERE
    TRANSKRIBUS_HTR_ID = int(htr_id_str)               # <-- FIX HERE
except ValueError:
    logging.error("Could not convert line detection or HTR IDs to integers.")
    sys.exit(1)

###############################################################################
# Parse Arguments
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the PDF-to-text pipeline using Transkribus.
    """
    parser = argparse.ArgumentParser(description="Transkribus PDF-to-text pipeline")

    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF file in data/pdfs/, e.g. 'type-1.pdf'"
    )

    return parser.parse_args()

###############################################################################
# Helper: Find existing runs
###############################################################################
def find_existing_runs_in_folder(model_folder: Path) -> List[int]:
    """
    Look for existing 'run_XX' directories in the given folder.
    Returns a list of run numbers found.
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
# Helper: Write JSON log
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in logs/ocr_img2txt/<model_name>/.
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
# Helper: Extract text from PAGE XML
###############################################################################
def extract_text_from_page_xml(xml_content: str) -> str:
    """
    Extract text lines from a PAGE XML string returned by Transkribus.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        logging.error("Could not parse PAGE XML. Returning empty text.")
        return ""

    ns = {'ns': root.tag.split('}')[0].strip('{')}
    lines = []
    for unicode_elem in root.findall('.//ns:Unicode', ns):
        if unicode_elem.text:
            lines.append(unicode_elem.text)
    return "\n".join(lines)

###############################################################################
# Main
###############################################################################
def main() -> None:
    """
    Main function for PDF-to-text pipeline using Transkribus.
    """
    # -------------------------------------------------------------------------
    # 1. Parse arguments and configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf
    model_name = "transkribus"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("Starting Transkribus PDF-to-text pipeline...")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"OCR model name: {model_name}")

    if not TRANSKRIBUS_USERNAME or not TRANSKRIBUS_PASSWORD:
        logging.error("TRANSKRIBUS_USERNAME/PASSWORD not set in .env or environment.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Convert PDF to TIFF images in data/page_by_page/<pdf_stem>/
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"Could not find PDF file: {pdf_path}")
        sys.exit(1)

    out_dir = DATA_DIR / "page_by_page" / pdf_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting PDF to TIFF images => {out_dir} ...")
    pages = convert_from_path(str(pdf_path))
    image_paths = []

    for i, page_img in enumerate(pages, start=1):
        # Save each page as a TIFF with 4-digit zero-padding
        img_name = f"page_{i:04}.tiff"  # <-- If needed: page_0001.tiff, etc.
        img_path = out_dir / img_name
        page_img.save(img_path, "TIFF")
        image_paths.append(img_path)

    logging.info(f"PDF split into {len(image_paths)} page images (TIFF).")

    # -------------------------------------------------------------------------
    # 3. Create results folder for this run:
    #    results/ocr_img2txt/transkribus/<pdf_stem>/run_XX/page_by_page/
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
    # 4. Call Transkribus OCR on all pages (handle None returns gracefully)
    # -------------------------------------------------------------------------
    logging.info("Calling Transkribus API...")
    try:
        # Open a session with your Transkribus credentials
        with transkribus_metagrapho_api(TRANSKRIBUS_USERNAME, TRANSKRIBUS_PASSWORD) as api:
            # This call returns a list of PAGE XML strings (or None if failed)
            all_page_xml = api(
                *image_paths,
                line_detection=TRANSKRIBUS_LINE_DETECTION_ID,
                htr_id=TRANSKRIBUS_HTR_ID
            )
    except Exception as e:
        logging.error(f"Transkribus OCR error: {e}")
        sys.exit(1)

    # Ensure we have something for each page (some may be None if 500 error)
    page_text_files = []
    for img_path, xml_content in zip(image_paths, all_page_xml):
        page_id = img_path.stem  # e.g., "page_0001"

        if xml_content is None:
            logging.error(f"Transkribus returned None (HTTP error) for {img_path}. Skipping page.")
            continue

        # Save PAGE XML
        xml_path = run_page_dir / f"{page_id}.xml"
        with open(xml_path, 'w', encoding='utf-8') as xf:
            xf.write(xml_content)

        # Extract text lines from the PAGE XML
        extracted_text = extract_text_from_page_xml(xml_content)
        if not extracted_text:
            logging.warning(f"No text extracted from {page_id}. PAGE XML may be empty or invalid.")

        # Save the extracted text
        out_txt_path = run_page_dir / f"{page_id}.txt"
        with open(out_txt_path, 'w', encoding='utf-8') as tf:
            tf.write(extracted_text)
        page_text_files.append(out_txt_path)

    logging.info("All pages processed (or skipped if errors). Individual text files created.")

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
        "pages_count": len(image_paths),
        "pages_success": len(page_text_files),
        "final_text_file": str(final_txt_path),
        "transkribus_line_detection_id": TRANSKRIBUS_LINE_DETECTION_ID,
        "transkribus_htr_id": TRANSKRIBUS_HTR_ID
    }

    write_json_log(log_info, model_name)
    logging.info("Run information successfully logged.")
    logging.info("Transkribus pipeline completed successfully. All done!")

###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    main()