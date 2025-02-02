#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# ---------------------------------------------------------------------
# OpenAI + requests for direct REST API calls:
# ---------------------------------------------------------------------
import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# Project Paths (same structure as Gemini code)
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
API_KEY = os.getenv("OPENAI_API_KEY")  # Replaces GOOGLE_API_KEY from Gemini

###############################################################################
# Argument Parsing
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for an OpenAI GPT-4 pdf-to-text pipeline.

    NOTE: Same arguments as Gemini—just replace references to Gemini with GPT-4.
    """
    parser = argparse.ArgumentParser(description="GPT-4 PDF-to-text pipeline")

    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Name of the PDF file in data/pdfs/, e.g. type-1.pdf"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the LLM call. Default: 0.0"
    )

    return parser.parse_args()

###############################################################################
# Utility: Find existing run_XY directories to auto-increment run number
###############################################################################
def find_existing_runs_in_temperature_folder(temp_folder: Path) -> List[int]:
    """
    Identical to Gemini code: scan for run_XX subfolders.
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
# Utility: Write a JSON log (same logic, just not calling it 'gemini-2.0')
###############################################################################
def write_json_log(log_dict: dict, model_name: str) -> None:
    """
    Save a JSON log file in logs/llm_img2txt/<model_name>/.

    The structure is identical—only the model name is now 'gpt-4o' (for example).
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
# OpenAI GPT-4 call: One-for-one replacement of the Gemini call
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
    # Helper to encode the PIL image as base64 PNG
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
            # Non-200 means some error from OpenAI
            raise ValueError(f"OpenAI error {response.status_code}: {response.text}")
    except Exception as e:
        logging.error(f"OpenAI GPT-4 call failed: {e}")
        return None, {}

###############################################################################
# Main Pipeline (Structure identical to gemini code)
###############################################################################
def main() -> None:
    """
    Main function for the “Gemini-style” pipeline, except with GPT‑4.
    """
    # 1. Parse arguments & configure logging
    args = parse_arguments()

    pdf_name = args.pdf
    temperature = args.temperature

    model_name = "gpt-4o"
    full_model_name = "gpt-4o-2024-08-06"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("Starting OpenAI GPT-4 PDF-to-text pipeline...")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model name: {model_name}")
    logging.info(f"Full model name: {full_model_name}")
    logging.info(f"Temperature: {temperature}")

    # 2. Load the transcription prompt
    #    If you prefer a different filename, just change "gpt-4o.txt".
    prompt_path = PROMPTS_DIR / f"{model_name}.txt"
    if not prompt_path.is_file():
        logging.error(f"Could not find prompt file: {prompt_path}")
        sys.exit(1)

    with open(prompt_path, 'r', encoding='utf-8') as pf:
        transcription_prompt = pf.read().strip()

    if not transcription_prompt:
        logging.error(f"The prompt file {prompt_path} is empty.")
        sys.exit(1)

    logging.info(f"Prompt loaded from: {prompt_path}")

    # 3. Convert PDF => page images in data/page_by_page/<pdf_stem>
    pdf_stem = Path(pdf_name).stem
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

    # 4. Create results folder:
    #    /Users/..../results/llm_img2txt/gpt-4o/<pdf_stem>/temperature_x.x/run_xy/page_by_page
    base_results_path = (
        Path("/Users/niclasgriesshaber/Desktop/llm_historical_dataset_pipeline_benchmarking")
        / "results" / "llm_img2txt" / model_name / pdf_stem
    )
    temp_dir = base_results_path / f"temperature_{temperature}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = find_existing_runs_in_temperature_folder(temp_dir)
    next_run_number = (max(existing_runs) + 1) if existing_runs else 1

    run_dir_name = f"run_{str(next_run_number).zfill(2)}"
    run_dir = temp_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=False)

    run_page_dir = run_dir / "page_by_page"
    run_page_dir.mkdir(parents=True, exist_ok=False)

    logging.info(f"Created run folder: {run_dir}")

    # 5. For each page image, call GPT-4 with infinite retry
    max_tokens = 16384  # Adjust if needed

    page_text_files = []
    total_prompt_tokens = 0
    total_candidates_tokens = 0  # We'll map this to 'completion_tokens'
    total_tokens = 0

    for img_path in image_paths:
        page_id = img_path.stem
        logging.info(f"Transcribing {page_id} ...")

        pil_image = Image.open(img_path)

        while True:
            try:
                text_out, usage_info = openai_api(
                    prompt=transcription_prompt,
                    pil_image=pil_image,
                    full_model_name=full_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=API_KEY
                )
                if text_out is None:
                    raise ValueError("Received None from openai_api")

                prompt_count = usage_info.get("prompt_tokens", 0)
                completion_count = usage_info.get("completion_tokens", 0)
                total_count = usage_info.get("total_tokens", 0)

                logging.info(
                    f"Usage for {page_id}: "
                    f"prompt_token_count={prompt_count}, "
                    f"candidates_token_count={completion_count}, "
                    f"total_token_count={total_count}"
                )

                total_prompt_tokens += prompt_count
                total_candidates_tokens += completion_count
                total_tokens += total_count

            except Exception as e:
                logging.error(f"OpenAI error: {e}. Retrying...")
                continue  # infinite retry
            else:
                break  # success => exit infinite loop

        # Save the text to page_xxxx.txt
        if not text_out:
            logging.warning(f"Empty response for {page_id}. Saving empty file.")
            text_out = ""

        out_txt_path = run_page_dir / f"{page_id}.txt"
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write(text_out)

        page_text_files.append(out_txt_path)

    # Summaries
    logging.info("All pages transcribed. Individual text files created.")
    logging.info(
        f"Total usage => prompt_tokens={total_prompt_tokens}, "
        f"candidates_tokens={total_candidates_tokens}, "
        f"total_tokens={total_tokens}"
    )

    # 6. Concatenate all page text files into <pdf_stem>.txt in the run folder
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            with open(txt_file, 'r', encoding='utf-8') as tf:
                outf.write(tf.read().strip())
                outf.write("\n\n")  # separate pages by blank line

    logging.info(f"Final concatenated file: {final_txt_path}")

    # 7. Write a JSON log summarizing the run
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "model_name": model_name,
        "full_model_name": full_model_name,
        "temperature": temperature,
        "run_directory": str(run_dir),
        "prompt_file": str(prompt_path),
        "pages_count": len(page_text_files),
        "final_text_file": str(final_txt_path),
        "total_usage": {
            "prompt_tokens": total_prompt_tokens,
            "candidates_tokens": total_candidates_tokens,
            "total_tokens": total_tokens
        }
    }

    write_json_log(log_info, model_name)
    logging.info("Run information successfully logged.")

    logging.info("Pipeline completed successfully with GPT‑4! All done.")


if __name__ == "__main__":
    main()