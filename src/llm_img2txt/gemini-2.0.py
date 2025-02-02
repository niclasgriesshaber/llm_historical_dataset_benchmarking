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

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

###############################################################################
# Project Paths
###############################################################################
# In your structure, "gemini-2.0.py" is in: project_root/src/llm_img2txt/gemini-2.0.py
# So .parents[2] should be the project root. Adjust if needed.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src" / "prompts" / "llm_img2txt"
RESULTS_DIR = PROJECT_ROOT / "results" / "llm_img2txt"
LOGS_DIR = PROJECT_ROOT / "logs" / "llm_img2txt"

# .env file is in config/.env
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load environment variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")  # Make sure this matches your .env key

###############################################################################
# Argument Parsing
###############################################################################
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the Gemini LLM PDF-to-text pipeline.

    NOTE: We removed the --model_name flag as requested.
    """
    parser = argparse.ArgumentParser(description="Gemini 2.0 PDF-to-text pipeline")

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
    Look for existing 'run_XX' directories in the temperature-specific folder.

    Args:
        temp_folder (Path): The folder to scan.

    Returns:
        List[int]: A list of run numbers (integers) found in the folder.
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
    Save a JSON log file in the logs directory.

    Args:
        log_dict (dict): The dictionary containing run metadata.
        model_name (str): The name of the model used (e.g., 'gemini-2.0').
    """
    # Create logs/<model_name>/llm_img2txt/ if it doesn't exist
    pipeline_logs_dir = LOGS_DIR / model_name
    pipeline_logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp_str}.json"
    log_path = pipeline_logs_dir / log_filename

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_dict, f, indent=4)

    logging.info(f"JSON log saved at: {log_path}")


###############################################################################
# Gemini 2.0 API Call (Unchanged)
###############################################################################
def gemini_api(
    prompt: str,
    pil_image: Image.Image,
    full_model_name: str,
    max_tokens: int,
    temperature: float,
    api_key: str
) -> Optional[str]:
    """
    Call Gemini 2.0 API to generate content based on a prompt and an image.

    NOTE: We are not changing this function (per your request). 
    Also note we no longer call this function from `main()`.
    """
    try:
        # Create a temporary PNG file from the PIL image to upload
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            pil_image.save(temp_file.name, "PNG")

            # Initialize Gemini client
            client = genai.Client(api_key=api_key)

            # Upload the temporary PNG file
            file_upload = client.files.upload(path=temp_file.name)

            # Generate content with config parameters (incorporating temperature)
            response = client.models.generate_content(
                model=full_model_name,
                contents=[
                    types.Part.from_uri(file_uri=file_upload.uri, mime_type=file_upload.mime_type),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=0.95,
                    top_k=20,
                    candidate_count=1,
                    seed=5,
                    max_output_tokens=max_tokens,
                    stop_sequences=["STOP!"],
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                )
            )
        return response.text
    except Exception as e:
        logging.error(f"Gemini 2.0 API error: {e}")
        return None


###############################################################################
# Main Pipeline
###############################################################################
def main() -> None:
    """
    Main function for Gemini 2.0 PDF-to-text pipeline.

    Steps:
      1. Parse arguments and configure logging.
      2. Load transcription prompt from prompts/llm_img2txt/gemini-2.0.txt.
      3. Convert PDF to page images and store them in data/page_by_page/<pdf_stem>/.
      4. Create results folder under:
         /Users/niclasgriesshaber/Desktop/llm_historical_dataset_pipeline_benchmarking/
         results/llm_img2txt/gemini-2.0/<pdf_stem>/temperature_x.x/run_xy/page_by_page/
      5. For each page image, call Gemini to get text (with infinite retry),
         log token usage, and save page_000X.txt.
      6. Concatenate all page text files into <pdf_stem>.txt.
      7. Write a JSON log in logs/llm_img2txt/gemini-2.0/.
    """
    # -------------------------------------------------------------------------
    # 1. Parse arguments and configure logging
    # -------------------------------------------------------------------------
    args = parse_arguments()

    pdf_name = args.pdf            # e.g., "type-1.pdf"
    temperature = args.temperature # default 0.0

    # Hardcode the short model name to "gemini-2.0" (per your request)
    model_name = "gemini-2.0"
    # Hardcode the full model name to the flash experimental version
    full_model_name = "gemini-2.0-flash-exp"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("Starting Gemini 2.0 PDF-to-text pipeline...")
    logging.info(f"PDF to process: {pdf_name}")
    logging.info(f"Model name: {model_name}")
    logging.info(f"Full model name: {full_model_name}")
    logging.info(f"Temperature: {temperature}")

    # -------------------------------------------------------------------------
    # 2. Load the transcription prompt from prompts/llm_img2txt/gemini-2.0.txt
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 3. Convert PDF to page images in data/page_by_page/<pdf_stem>/
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
    # 4. Create the results folder under:
    #    /Users/niclasgriesshaber/Desktop/llm_historical_dataset_pipeline_benchmarking/
    #    results/llm_img2txt/gemini-2.0/<pdf_stem>/temperature_x.x/run_xy/page_by_page/
    # -------------------------------------------------------------------------
    base_results_path = (
        Path("/Users/niclasgriesshaber/Desktop/llm_historical_dataset_pipeline_benchmarking")
        / "results" / "llm_img2txt" / model_name / pdf_stem
    )

    # e.g. /.../gemini-2.0/type-1/temperature_0.0
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

    # -------------------------------------------------------------------------
    # 5. For each page image, call Gemini with infinite retry + usage logging
    # -------------------------------------------------------------------------
    max_tokens = 8192  # Adjust as needed

    page_text_files = []

    # We'll track total token usage for the entire PDF
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0

    for img_path in image_paths:
        page_id = img_path.stem  # e.g. "page_0001"
        logging.info(f"Transcribing {page_id} ...")

        # Load the PIL image
        pil_image = Image.open(img_path)

        # =====================================================================
        # Infinite retry block: We do the API call right here, 
        # exactly as recommended for usage metadata
        # =====================================================================
        while True:
            try:
                # Create a temporary PNG file from the PIL image to upload
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    pil_image.save(temp_file.name, "PNG")

                    # Initialize Gemini client
                    client = genai.Client(api_key=API_KEY)

                    # Upload the temporary PNG file
                    file_upload = client.files.upload(path=temp_file.name)

                    # Actually call generate_content
                    response = client.models.generate_content(
                        model=full_model_name,
                        contents=[
                            types.Part.from_uri(
                                file_uri=file_upload.uri,
                                mime_type=file_upload.mime_type
                            ),
                            transcription_prompt
                        ],
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            top_p=0.95,
                            top_k=20,
                            candidate_count=1,
                            seed=5,
                            max_output_tokens=max_tokens,
                            stop_sequences=["STOP!"],
                            presence_penalty=0.0,
                            frequency_penalty=0.0,
                        )
                    )
                transcription = response.text

                # usage_metadata is a GenerateContentResponseUsageMetadata object
                usage = response.usage_metadata  # Not a dict, but a pydantic model

                # Print usage (page-by-page) to command line
                logging.info(
                    f"Usage for {page_id}: "
                    f"prompt_token_count={usage.prompt_token_count}, "
                    f"candidates_token_count={usage.candidates_token_count}, "
                    f"total_token_count={usage.total_token_count}"
                )

                # Accumulate usage
                if usage.prompt_token_count is not None:
                    total_prompt_tokens += usage.prompt_token_count
                if usage.candidates_token_count is not None:
                    total_candidates_tokens += usage.candidates_token_count
                if usage.total_token_count is not None:
                    total_tokens += usage.total_token_count

            except Exception as e:
                logging.error(f"Gemini 2.0 API error: {e}. Retrying...")
                continue  # retry indefinitely
            else:
                # If no exception, break out of the retry loop
                break

        # If transcription is empty or None, log a warning
        if not transcription:
            logging.warning(f"Received empty response for {page_id}. Saving empty file.")
            transcription = ""

        # Save page_xxxx.txt
        out_txt_path = run_page_dir / f"{page_id}.txt"
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        page_text_files.append(out_txt_path)

    logging.info("All pages transcribed. Individual text files created.")

    # Log total usage for the entire PDF
    logging.info(
        f"Total usage across all pages => "
        f"prompt_tokens={total_prompt_tokens}, "
        f"candidates_tokens={total_candidates_tokens}, "
        f"total_tokens={total_tokens}"
    )

    # -------------------------------------------------------------------------
    # 6. Concatenate all page text files into <pdf_stem>.txt in the run folder
    # -------------------------------------------------------------------------
    final_txt_path = run_dir / f"{pdf_stem}.txt"
    logging.info(f"Combining page texts into {final_txt_path} ...")
    with open(final_txt_path, 'w', encoding='utf-8') as outf:
        for txt_file in sorted(page_text_files):
            with open(txt_file, 'r', encoding='utf-8') as tf:
                outf.write(tf.read().strip())
                outf.write("\n\n")  # Separate pages by blank line, if desired

    logging.info(f"Final concatenated file: {final_txt_path}")

    # -------------------------------------------------------------------------
    # 7. Write a JSON log summarizing the run
    # -------------------------------------------------------------------------
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

    logging.info("Pipeline completed successfully. All done!")


###############################################################################
# Entry Point
###############################################################################
if __name__ == "__main__":
    main()