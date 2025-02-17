#!/usr/bin/env python3
"""
GPT-4o PDF -> JSON -> CSV (Single Request, Page-by-Page) Pipeline

Usage:
  python3 gpt-4o.py --pdf <your_pdf_name.pdf>

This script:
1. Loads a PDF from 'data/pdfs/' using PyMuPDFLoader (via langchain_community).
2. Reads a corresponding prompt from 'src/prompts/llm_pdf2csv/gpt-4o.txt'.
3. Calls the gpt-4o model page by page to produce structured JSON.
4. Parses and cleans that JSON into final CSV output (with an 'id' column).
5. Saves artifacts (JSON, CSV) in a dedicated run folder under 'results/llm_pdf2csv'.
6. Logs usage data (basic) and total runtime for reproducibility.
"""

import os
import sys
import re
import json
import time
import argparse
import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

# LangChain + community loaders and models
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Load environment variables (OPENAI_API_KEY)
from dotenv import load_dotenv
load_dotenv()

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_pdf2csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_pdf2csv"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_pdf2csv"

###############################################################################
# Model constants
###############################################################################
MODEL_NAME = "gpt-4o"          # For folder naming
FULL_MODEL_NAME = "gpt-4o"     # The actual model string used in ChatOpenAI
MAX_OUTPUT_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0

###############################################################################
# Utility: Time formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a time duration in seconds into a standard 'H:MM:SS' string for clearer logging output.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

###############################################################################
# Utility: Reorder dictionary keys with page_number at the end (optional)
###############################################################################
def reorder_dict_with_page_number(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a new dictionary ensuring 'page_number' and 'additional_information'
    appear last if present. Helps maintain consistent CSV column ordering.
    """
    special_keys = {"page_number", "additional_information"}
    base_keys = [k for k in d.keys() if k not in special_keys]
    base_keys.sort()

    out = {}
    for k in base_keys:
        out[k] = d[k]

    if "page_number" in d:
        out["page_number"] = d["page_number"]
    if "additional_information" in d:
        out["additional_information"] = d["additional_information"]

    return out

###############################################################################
# Utility: Convert JSON to CSV (with an 'id' column)
###############################################################################
def convert_json_to_csv(json_data: Union[Dict, List], csv_path: Path) -> None:
    """
    Takes JSON data (dict or list of dicts) and writes it to a CSV file at csv_path.
    Reorders keys so 'page_number' + 'additional_information' appear last, then
    adds 'id' as the final column.
    """
    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        # Fallback: single row with "value"
        records = [{"value": str(json_data)}]

    # Gather all keys
    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    special_order = ["page_number", "additional_information"]
    base_keys = [k for k in all_keys if k not in special_order]
    base_keys.sort()
    ordered_special = [k for k in special_order if k in all_keys]
    fieldnames = base_keys + ordered_special + ["id"]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row_id = 1
        for rec in records:
            if not isinstance(rec, dict):
                # If rec is a plain value, store in "value" col
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                row_data["id"] = row_id
                writer.writerow(row_data)
                row_id += 1
                continue

            row_data = {}
            for fn in fieldnames:
                if fn == "id":
                    row_data[fn] = row_id
                else:
                    row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)
            row_id += 1

###############################################################################
# Utility: Clean & Parse JSON from the LLM's response text
###############################################################################
def clean_and_parse_json(response_text: str) -> Dict[str, Any]:
    """
    Attempts to extract a JSON dictionary from the model's response text.
    Some LLMs may wrap the JSON in code fences or prefix it with 'json' or backticks.
    We'll strip those out and parse it.

    Raises ValueError if it cannot parse a valid JSON structure.
    """
    # Remove backticks (```...``` or `...`)
    cleaned = response_text.replace("```", "").replace("`", "")
    # If the model sometimes includes 'json' literal in code fence
    cleaned = cleaned.replace("json", "")
    # Attempt to parse
    return json.loads(cleaned.strip())

###############################################################################
# Main Execution
###############################################################################
def main():
    """
    1. Parse arguments (which PDF to process, plus optional temperature).
    2. Load PDF pages via PyMuPDFLoader.
    3. Load a text prompt from 'src/prompts/llm_pdf2csv/gpt-4o.txt'.
    4. For each page, call ChatOpenAI(gpt-4o) with the LLMChain to extract structured JSON.
    5. Reorder + combine all pages' results -> single list.
    6. Save JSON and CSV to a new run directory.
    7. Log the run.
    """
    parser = argparse.ArgumentParser(description="GPT-4o PDF->JSON->CSV Pipeline (page-by-page)")
    parser.add_argument("--pdf", required=True, help="Name of PDF in data/pdfs/, e.g. my_file.pdf")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for GPT-4o calls")
    args = parser.parse_args()

    pdf_name = args.pdf
    temperature = args.temperature

    # Configure logging (console)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("=== GPT-4o PDF->JSON->CSV Pipeline (LangChain) ===")
    logging.info(f"PDF: {pdf_name} | Temperature: {temperature}")

    # Verify PDF path
    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"PDF not found: {pdf_path}")
        sys.exit(1)
    pdf_stem = pdf_path.stem

    # Load prompt text
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Missing prompt file: {prompt_path}")
        sys.exit(1)
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        logging.error(f"Prompt file is empty: {prompt_path}")
        sys.exit(1)

    # Prepare results folder => results/llm_pdf2csv/gpt-4o/<pdf_name>/temperature_x.x/run_nn/
    pdf_folder = RESULTS_DIR / MODEL_NAME / pdf_name
    temp_folder = pdf_folder / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Identify next run_xx
    existing_runs = []
    for child in temp_folder.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                run_num = int(child.name.split("_")[1])
                existing_runs.append(run_num)
            except ValueError:
                pass
    next_run = max(existing_runs) + 1 if existing_runs else 1
    run_folder = temp_folder / f"run_{str(next_run).zfill(2)}"
    run_folder.mkdir(parents=True, exist_ok=False)

    final_csv_path = run_folder / f"{pdf_stem}.csv"
    pdf_json_path = run_folder / f"{pdf_stem}.json"

    overall_start_time = time.time()

    # --------------------- Setup the LLM & prompt chain -----------------------
    # Use the loaded prompt as a template
    prompt = PromptTemplate(template=prompt_text, input_variables=["doc_text"])
    llm = ChatOpenAI(model=FULL_MODEL_NAME, temperature=temperature)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # --------------------- Load PDF Pages (PyMuPDFLoader) ---------------------
    loader = PyMuPDFLoader(str(pdf_path))
    doc = loader.load()  # doc is a list of Document pages

    # We store all page results in a list
    combined_results = []

    # For each page, we send the text to LLMChain
    for page_index, page in enumerate(doc, start=1):
        text = page.page_content
        logging.info(f"Processing page {page_index} with ~{len(text)} characters...")

        # Invoke the chain
        response = llm_chain.invoke({"doc_text": text})
        llm_text = response["text"]  # The raw text from LLM

        # Clean and parse JSON
        try:
            data_dict = clean_and_parse_json(llm_text)
        except ValueError as ve:
            logging.error(f"Failed to parse JSON on page {page_index}. Raw response:\n{llm_text}")
            logging.error(str(ve))
            data_dict = {}

        # Optionally add "page_number" for reference
        data_dict["page_number"] = page_index

        # Reorder special keys
        data_dict = reorder_dict_with_page_number(data_dict)
        combined_results.append(data_dict)

    # -------------- Write combined JSON --------------
    with pdf_json_path.open("w", encoding="utf-8") as jf:
        json.dump(combined_results, jf, indent=2, ensure_ascii=False)
    logging.info(f"Saved JSON to {pdf_json_path}")

    # -------------- Convert to CSV --------------
    convert_json_to_csv(combined_results, final_csv_path)
    logging.info(f"Saved CSV to {final_csv_path}")

    # -------------- Write a run log (JSON) --------------
    total_duration = time.time() - overall_start_time
    # (We do not currently have token usage from ChatOpenAI in this snippet)

    log_path = LOGS_DIR / MODEL_NAME / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": MODEL_NAME,
        "full_model_name": FULL_MODEL_NAME,
        "temperature_passed_in": temperature,
        "run_directory": str(run_folder),
        "prompt_file": str(prompt_path),
        "json_file": str(pdf_json_path),
        "csv_file": str(final_csv_path),
        "total_duration_seconds": int(total_duration),
        "total_duration_formatted": format_duration(total_duration),
    }
    with log_path.open("w", encoding="utf-8") as lf:
        json.dump(log_data, lf, indent=4)

    logging.info(f"Pipeline completed in {format_duration(total_duration)}. All done!")

if __name__ == "__main__":
    main()