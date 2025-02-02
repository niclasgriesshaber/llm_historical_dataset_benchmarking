#!/usr/bin/env python3

import os
import sys
import re
import json
import argparse
import logging
import tempfile
import requests  # For OpenAI API calls
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Dict, List, Optional

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# OpenAI GPT-4 call: from your instructions
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
    Call OpenAI’s GPT-4 with an image + text prompt.

    We do a simple approach:
      1) Convert the PIL image to base64 PNG.
      2) Pass prompt + the image as a 'user' message with an array of blocks.
      3) Return the text plus usage stats in a dict (prompt_tokens, completion_tokens, total_tokens).

    If there's an error, we return (None, {}).
    """
    import base64
    from io import BytesIO

    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    with BytesIO() as buffer:
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": full_model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
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
            raise ValueError(f"OpenAI error {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"OpenAI GPT-4 call failed: {e}")
        return None, {}

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # e.g. /.../llm_historical_dataset_pipeline_benchmarking
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_img2csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_img2csv"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_img2csv"
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model constants (adjust as needed)
MODEL_NAME = "gpt-4o"         # Short name for folder naming
FULL_MODEL_NAME = "gpt-4o-2024-08-06"    # OpenAI's GPT-4 model ID
MAX_OUTPUT_TOKENS = 16134     # This is "max_tokens" in the openai_api call

###############################################################################
# Utility: Parse JSON from LLM text (strip out triple backticks if present)
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON (```json ... ```), or fallback to raw text.
    Then parse it as JSON.

    Raises ValueError if parsing fails.
    """
    fenced_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text, re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        # fallback to entire response, removing any backticks
        candidate = response_text.strip().strip('`')

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as ex:
        raise ValueError(
            f"Failed to parse JSON. Error: {ex}\nFull response text:\n{candidate}"
        )

###############################################################################
# Utility: Reorder dictionary keys with page_number at the end
###############################################################################
def reorder_dict_with_page_number(d: Dict[str, Any], page_number: int) -> Dict[str, Any]:
    """
    Return a new dictionary that includes a 'page_number' key
    and places 'additional_information' at the end,
    preserving other keys in alphabetical order.
    """
    special_keys = {"page_number", "additional_information"}
    base_keys = [k for k in d.keys() if k not in special_keys]

    out = {}
    for k in base_keys:
        out[k] = d[k]

    out["page_number"] = page_number

    if "additional_information" in d:
        out["additional_information"] = d["additional_information"]

    return out

###############################################################################
# Utility: Convert array of JSON objects to a single CSV
###############################################################################
def convert_json_to_csv(json_data: Union[Dict, List], csv_path: Path) -> None:
    """
    Flatten JSON objects/arrays into CSV.
    1) If top-level is a single dict, that's 1 row.
    2) If top-level is a list, each element is 1 row.
    3) Reorder columns so that 'page_number' and 'additional_information' come last.
    """
    import csv

    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        # Fallback: treat as a single row with "value" column
        records = [{"value": str(json_data)}]

    # Gather all keys
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
                # If for some reason we got a raw str
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                writer.writerow(row_data)
                continue

            row_data = {}
            for fn in fieldnames:
                row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)

###############################################################################
# OpenAI GPT-4 API Call with Infinite Retry
###############################################################################
def gpt4_api_call(
    prompt: str,
    pil_image: Image.Image,
    temperature: float
) -> Optional[dict]:
    """
    Call GPT-4 with the given prompt + image, infinitely retry on failure.
    Returns a dict:
        {
          "text": <the text response>,
          "usage": <usage metadata object with {prompt_tokens, completion_tokens, total_tokens}>
        }
    """
    while True:
        try:
            text_out, usage_info = openai_api(
                prompt=prompt,
                pil_image=pil_image,
                full_model_name=FULL_MODEL_NAME,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=temperature,
                api_key=OPENAI_API_KEY
            )

            if not text_out:
                logging.error("GPT-4 returned None or empty text. Retrying...")
                continue

            return {
                "text": text_out,
                "usage": usage_info
            }

        except Exception as e:
            logging.error(f"OpenAI GPT-4 call failed: {e}. Retrying...")

###############################################################################
# Main
###############################################################################
def main():
    # -------------------------------------------------------------------------
    # Parse command-line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="OpenAI GPT-4 PDF-to-JSON-to-CSV pipeline")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Name of the PDF in data/pdfs/, e.g. type-1.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default = 0.0)"
    )
    args = parser.parse_args()

    pdf_name = args.pdf
    temperature = args.temperature

    # -------------------------------------------------------------------------
    # Configure logging
    # -------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("Starting llm_img2csv pipeline with OpenAI GPT-4...")
    logging.info(f"PDF: {pdf_name}, Temperature: {temperature}")

    pdf_path = DATA_DIR / "pdfs" / pdf_name
    if not pdf_path.is_file():
        logging.error(f"Could not find PDF: {pdf_path}")
        sys.exit(1)

    # We'll still use the stem for naming PNG directories, etc.
    pdf_stem = Path(pdf_name).stem

    # -------------------------------------------------------------------------
    # Load the task prompt from prompts/llm_img2csv/<MODEL_NAME>.txt
    # -------------------------------------------------------------------------
    prompt_path = PROMPTS_DIR / f"{MODEL_NAME}.txt"
    if not prompt_path.is_file():
        logging.error(f"Missing prompt file: {prompt_path}")
        sys.exit(1)

    task_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not task_prompt:
        logging.error(f"Empty prompt file: {prompt_path}")
        sys.exit(1)

    logging.info(f"Loaded task prompt from: {prompt_path}")

    # -------------------------------------------------------------------------
    # Check / create data/page_by_page/<pdf_stem> for PNG generation
    # -------------------------------------------------------------------------
    page_by_page_dir = DATA_DIR / "page_by_page" / pdf_stem
    if not page_by_page_dir.is_dir():
        logging.info(f"No existing page-by-page folder found; converting PDF to PNG...")
        page_by_page_dir.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(pdf_path))
        for i, pil_img in enumerate(pages, start=1):
            out_png = page_by_page_dir / f"page_{i:04d}.png"
            pil_img.save(out_png, "PNG")
        logging.info(f"Created {len(pages)} PNG pages in {page_by_page_dir}")

    else:
        logging.info(f"Folder {page_by_page_dir} already exists; skipping PDF->PNG step.")

    # Gather sorted PNGs
    png_files = sorted(page_by_page_dir.glob("page_*.png"))
    if not png_files:
        logging.error(f"No PNG pages found in {page_by_page_dir}. Exiting.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Prepare results folder => .../llm_img2csv/gpt-4/<pdf_name>/temperature_x.x/run_xy/
    # -------------------------------------------------------------------------
    pdf_folder = RESULTS_DIR / MODEL_NAME / pdf_name
    temp_folder = pdf_folder / f"temperature_{temperature}"
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Find existing runs
    existing_runs = []
    for child in temp_folder.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                run_num = int(child.name.split("_")[1])
                existing_runs.append(run_num)
            except ValueError:
                pass

    next_run = max(existing_runs) + 1 if existing_runs else 1
    run_folder_name = f"run_{str(next_run).zfill(2)}"
    run_folder = temp_folder / run_folder_name
    run_folder.mkdir(parents=True, exist_ok=False)
    logging.info(f"Run folder: {run_folder}")

    # Subfolder for page-by-page JSON
    run_page_json_dir = run_folder / "page_by_page"
    run_page_json_dir.mkdir(parents=True, exist_ok=False)

    # The final CSV will be in: run_folder/<pdf_name>.csv
    final_csv_path = run_folder / f"{pdf_name}.csv"

    # -------------------------------------------------------------------------
    # Accumulators for usage tokens and final data
    # -------------------------------------------------------------------------
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    merged_data: List[Any] = []

    # -------------------------------------------------------------------------
    # Process each page with infinite retry on GPT-4
    # -------------------------------------------------------------------------
    for idx, png_path in enumerate(png_files, start=1):
        page_id = png_path.stem  # e.g. "page_0001"
        logging.info(f"Processing {page_id} ...")

        # Load image
        try:
            pil_image = Image.open(png_path)
        except Exception as e:
            logging.error(f"Cannot open image {png_path}: {e}")
            continue

        # Infinite retry call
        result = gpt4_api_call(
            prompt=task_prompt,
            pil_image=pil_image,
            temperature=temperature
        )
        if result is None:
            # If somehow we still get None, just skip (but we have infinite retry so unlikely)
            logging.error(f"OpenAI GPT-4 returned None for {page_id}, skipping.")
            continue

        response_text = result["text"]
        usage_meta = result["usage"]

        # Update usage accumulators
        if usage_meta:
            ptc = usage_meta.get("prompt_tokens", 0)
            ctc = usage_meta.get("completion_tokens", 0)
            ttc = usage_meta.get("total_tokens", ptc + ctc)
            total_prompt_tokens += ptc
            total_completion_tokens += ctc
            total_tokens += ttc

            logging.info(
                f"Usage for {page_id} => prompt={ptc}, completion={ctc}, total={ttc} "
                f"(cumulative total={total_tokens})"
            )

        # Parse JSON with infinite retry
        while True:
            try:
                parsed = parse_json_str(response_text)
            except ValueError as ve:
                logging.error(f"JSON parse error for {page_id}: {ve}\nRetrying GPT-4 call...")
                # Re-call GPT-4
                new_result = gpt4_api_call(
                    prompt=task_prompt,
                    pil_image=pil_image,
                    temperature=temperature
                )
                if not new_result:
                    logging.error("GPT-4 returned None again, continuing infinite retry...")
                    continue
                response_text = new_result["text"]
                usage_meta = new_result["usage"]
                # Update usage accumulators again
                if usage_meta:
                    ptc = usage_meta.get("prompt_tokens", 0)
                    ctc = usage_meta.get("completion_tokens", 0)
                    ttc = usage_meta.get("total_tokens", ptc + ctc)
                    total_prompt_tokens += ptc
                    total_completion_tokens += ctc
                    total_tokens += ttc
                    logging.info(f"[Retry usage] total tokens so far: {total_tokens}")
            else:
                # No exception => break out of parse retry
                break

        # Save page-level JSON
        page_json_path = run_page_json_dir / f"{page_id}.json"
        with page_json_path.open("w", encoding="utf-8") as jf:
            json.dump(parsed, jf, indent=2, ensure_ascii=False)

        # Reorder for final CSV
        if isinstance(parsed, list):
            for obj in parsed:
                if isinstance(obj, dict):
                    merged_data.append(reorder_dict_with_page_number(obj, idx))
                else:
                    merged_data.append(obj)
        elif isinstance(parsed, dict):
            merged_data.append(reorder_dict_with_page_number(parsed, idx))
        else:
            merged_data.append(parsed)

    # -------------------------------------------------------------------------
    # Convert merged data to CSV
    # -------------------------------------------------------------------------
    convert_json_to_csv(merged_data, final_csv_path)
    logging.info(f"Final CSV saved at: {final_csv_path}")

    # -------------------------------------------------------------------------
    # Write JSON log with a timestamp-based filename in logs/llm_img2csv/gpt-4/
    # -------------------------------------------------------------------------
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
        "pages_count": len(png_files),
        "page_json_directory": str(run_page_json_dir),
        "final_csv": str(final_csv_path),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens
        }
    }

    with open(log_path, "w", encoding="utf-8") as lf:
        json.dump(log_data, lf, indent=4)

    logging.info(f"Run log saved at: {log_path}")
    logging.info("Pipeline completed successfully. All done!")


if __name__ == "__main__":
    main()