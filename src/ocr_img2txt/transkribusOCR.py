#!/usr/bin/env python3
"""
Transkribus PDF -> TIFF -> TEXT Pipeline with lxml-based PAGE XML parsing.

Requires:
  - pdf2image
  - Pillow (PIL)
  - python-dotenv
  - transkribus_metagrapho_api
  - lxml
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
from dotenv import load_dotenv
from transkribus_metagrapho_api import transkribus_metagrapho_api

# Import lxml for improved PAGE XML parsing
from lxml import etree

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

line_det_str = os.getenv("TRANSKRIBUS_LINE_DETECTION_ID")
htr_id_str = os.getenv("TRANSKRIBUS_HTR_ID")

if not line_det_str or not htr_id_str:
    print("Missing TRANSKRIBUS_LINE_DETECTION_ID or TRANSKRIBUS_HTR_ID in .env.")
    sys.exit(1)

try:
    TRANSKRIBUS_LINE_DETECTION_ID = int(line_det_str)
    TRANSKRIBUS_HTR_ID = int(htr_id_str)
except ValueError:
    print("Could not convert line detection or HTR IDs to integers.")
    sys.exit(1)

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
    Parse command-line arguments for the PDF-to-text pipeline using Transkribus.
    """
    parser = argparse.ArgumentParser(description="Transkribus PDF-to-text pipeline")
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF file in data/pdfs/, e.g. example.pdf"
    )
    return parser.parse_args()

###############################################################################
# Utility: Find existing runs in folder
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
# Utility: Extract text from PAGE XML with reading order (lxml-based)
###############################################################################
def extract_text_from_page_xml(xml_content: str) -> str:
    """
    Extract text from a PAGE XML string with reading order awareness.
    This function:
      - Finds the reading order (<ReadingOrder>) if present.
      - Iterates over TextRegion elements in that order.
      - For each TextLine, assembles text from <Word> elements if they exist.
        Otherwise, falls back to the line-level <TextEquiv>.
      - Joins lines with line breaks and regions with blank lines.
    """
    try:
        root = etree.fromstring(xml_content.encode("utf-8"))
    except etree.XMLSyntaxError:
        # Return empty string if parsing fails
        logging.error("Could not parse PAGE XML. Returning empty text.")
        return ""

    # Default PAGE namespace might look like:
    # "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    # We'll call it 'pc' for convenience.
    page_ns = root.nsmap.get(None, "")
    ns = {"pc": page_ns}

    # 1) Check <ReadingOrder>
    ro_xpath = ".//pc:ReadingOrder/pc:OrderedGroup/pc:RegionRefIndexed"
    region_ref_elements = root.findall(ro_xpath, namespaces=ns)

    if not region_ref_elements:
        # Fallback: retrieve TextRegions in document order
        text_regions = root.findall(".//pc:TextRegion", namespaces=ns)
    else:
        # Build a mapping of regionID -> reading order index
        region_order_map = {}
        for ref in region_ref_elements:
            region_id = ref.get("regionRef")
            index_str = ref.get("index")
            if region_id and index_str is not None:
                try:
                    region_order_map[region_id] = int(index_str)
                except ValueError:
                    pass

        # Now find all <TextRegion> and keep only those referenced
        all_regions = root.findall(".//pc:TextRegion", namespaces=ns)
        text_regions = []
        for region in all_regions:
            r_id = region.get("id")
            if r_id in region_order_map:
                text_regions.append(region)

        # Sort by reading order index
        text_regions.sort(key=lambda r: region_order_map[r.get("id")])

    all_regions_text = []

    # 2) Iterate over TextRegion in correct order
    for region in text_regions:
        region_lines = []
        text_lines = region.findall(".//pc:TextLine", namespaces=ns)

        for line in text_lines:
            # Word-level approach
            word_elements = line.findall(".//pc:Word", namespaces=ns)
            if word_elements:
                words = []
                for w_el in word_elements:
                    w_text_equiv = w_el.find(".//pc:TextEquiv", namespaces=ns)
                    if w_text_equiv is not None:
                        w_unicode = w_text_equiv.find("pc:Unicode", namespaces=ns)
                        if w_unicode is not None and w_unicode.text:
                            words.append(w_unicode.text.strip())
                line_text = " ".join(words)
                if line_text:
                    region_lines.append(line_text)
                continue

            # If no <Word>, fallback to line-level <TextEquiv>
            text_equiv = line.find(".//pc:TextEquiv", namespaces=ns)
            if text_equiv is not None:
                unicode_elem = text_equiv.find("pc:Unicode", namespaces=ns)
                if unicode_elem is not None and unicode_elem.text:
                    region_lines.append(unicode_elem.text.strip())

        if region_lines:
            region_block_text = "\n".join(region_lines)
            all_regions_text.append(region_block_text)

    # Join regions with double newlines
    return "\n\n".join(all_regions_text)

###############################################################################
# Main Pipeline
###############################################################################
def main() -> None:
    """
    Main function for PDF-to-text pipeline using Transkribus:

    Steps:
      1. Parse arguments & configure logging.
      2. Convert PDF => TIFF in data/page_by_page/TIFF/<pdf_stem>.
      3. Create results folder under results/ocr_img2txt/transkribus/<pdf_stem>/run_xy/page_by_page/.
      4. Perform OCR via Transkribus for each page (batch call).
      5. Parse PAGE XML with reading order to extract text (lxml).
      6. Concatenate all page text files into <pdf_stem>.txt.
      7. Write a JSON log in logs/ocr_img2txt/transkribus/.
      8. Print final summary.
    """
    # -------------------------------------------------------------------------
    # 1. Parse arguments and configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()
    pdf_name = args.pdf
    model_name = "transkribus"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== Transkribus PDF -> TIFF -> TEXT Pipeline ===")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model: {model_name}")

    if not TRANSKRIBUS_USERNAME or not TRANSKRIBUS_PASSWORD:
        logging.error("TRANSKRIBUS_USERNAME/PASSWORD not set in .env or environment.")
        sys.exit(1)

    overall_start_time = time.time()

    # -------------------------------------------------------------------------
    # 2. Convert PDF to TIFF images in data/page_by_page/TIFF/<pdf_stem>
    #    (Skip if folder already exists and is non-empty)
    # -------------------------------------------------------------------------
    pdf_stem = Path(pdf_name).stem
    pdf_path = PROJECT_ROOT / "data" / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"Could not find PDF file: {pdf_path}")
        sys.exit(1)

    tiff_dir = PROJECT_ROOT / "data" / "page_by_page" / "TIFF" / pdf_stem
    if not tiff_dir.is_dir():
        logging.info(f"No TIFF folder found; converting PDF -> TIFF in {tiff_dir} ...")
        tiff_dir.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(pdf_path))
        for i, page_img in enumerate(pages, start=1):
            img_path = tiff_dir / f"page_{i:04d}.tiff"
            page_img.save(img_path, "TIFF")
        logging.info(f"Created {len(pages)} TIFF pages in {tiff_dir}")
    else:
        logging.info(f"Folder {tiff_dir} already exists; skipping PDF->TIFF step.")

    tiff_files = sorted(tiff_dir.glob("page_*.tiff"))
    if not tiff_files:
        logging.error(f"No page images found in {tiff_dir}. Exiting.")
        sys.exit(1)

    total_pages = len(tiff_files)

    # -------------------------------------------------------------------------
    # 3. Create the results folder:
    #    results/ocr_img2txt/transkribus/<pdf_stem>/run_xy/page_by_page/
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
    # 4. Call Transkribus OCR on all pages in a single batch
    # -------------------------------------------------------------------------
    logging.info("Calling Transkribus API for all pages...")
    page_text_files = []
    try:
        with transkribus_metagrapho_api(TRANSKRIBUS_USERNAME, TRANSKRIBUS_PASSWORD) as api:
            all_page_xml = api(
                *tiff_files,
                htr_id=TRANSKRIBUS_HTR_ID
            )
    except Exception as e:
        logging.error(f"Transkribus OCR error: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 5. For each page's XML, parse text with reading order and save
    # -------------------------------------------------------------------------
    for idx, (tiff_path, xml_content) in enumerate(zip(tiff_files, all_page_xml), start=1):
        logging.info(f"Processing page {idx} of {total_pages}: {tiff_path.name}")
        page_id = tiff_path.stem  # e.g., "page_0001"

        if xml_content is None:
            logging.error(f"Transkribus returned None (HTTP error) for {tiff_path}. Skipping page.")
            continue

        # Save PAGE XML
        xml_path = run_page_dir / f"{page_id}.xml"
        xml_path.write_text(xml_content, encoding='utf-8')

        # Extract text (lxml-based function)
        extracted_text = extract_text_from_page_xml(xml_content)
        if not extracted_text:
            logging.warning(f"No text extracted for {tiff_path.stem}. PAGE XML may be empty or invalid.")

        # Save text
        out_txt_path = run_page_dir / f"{page_id}.txt"
        out_txt_path.write_text(extracted_text, encoding='utf-8')
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
    # 6. Concatenate all page text files into <pdf_stem>.txt
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            text_content = txt_file.read_text(encoding='utf-8').strip()
            outf.write(text_content + "\n\n")

    logging.info(f"Final concatenated file: {final_txt_path}")

    # -------------------------------------------------------------------------
    # 7. Write a JSON log summarizing the run
    # -------------------------------------------------------------------------
    total_duration = time.time() - overall_start_time
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": model_name,
        "run_directory": str(run_dir),
        "pages_count": total_pages,
        "pages_successfully_processed": len(page_text_files),
        "final_text_file": str(final_txt_path),
        "transkribus_line_detection_id": TRANSKRIBUS_LINE_DETECTION_ID,
        "transkribus_htr_id": TRANSKRIBUS_HTR_ID,
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration)
    }

    write_json_log(log_info, model_name)

    # -------------------------------------------------------------------------
    # 8. Final summary
    # -------------------------------------------------------------------------
    logging.info("Run information successfully logged.")
    logging.info(
        f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
    )
    logging.info("All done!")

###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    main()