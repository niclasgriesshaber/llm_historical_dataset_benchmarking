#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmarking OCR vs. LLM for text extraction, parallelized with joblib + rapidfuzz.

Generates TWO HTML tables in one file:

1) Normalized Results Table
   - Non-ASCII removed entirely
   - Lowercase
   - Remove punctuation => only [a-z0-9] plus spaces
   - Collapse multiple spaces
   - Strip leading/trailing
   - Remove line breaks/tabs

2) Non-normalized Results Table
   - Preserve punctuation, casing, accented letters
   - Remove line breaks/tabs
   - Collapse multiple spaces
   - Strip leading/trailing

Each cell in both tables has 4 lines:
   1) Levenshtein distance
   2) ground-truth doc length (for that table's version)
   3) CER% (distance / length_of_that_version)
   4) WER%

Docs are sorted numerically so type-1 < type-2 < ... < type-9 < type-10, etc.
"""

import os
import re
import glob
import argparse
import logging
from joblib import Parallel, delayed
from rapidfuzz import distance

# ----------------- Configure Logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute OCR vs. LLM metrics (parallelized, rapidfuzz) for BOTH normalized and non-normalized text."
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0.0",
        help="Temperature folder name (default: 0.0) for LLM-based results."
    )
    parser.add_argument(
        "--run",
        type=str,
        default="run_01",
        help="Run folder name (default: run_01)."
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="llm_vs_ocr_benchmarking_results.html",
        help="Output HTML file (default: llm_vs_ocr_benchmarking_results.html)."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 = use all cores)."
    )
    return parser.parse_args()


def clean_text_nonorm(text):
    """
    Minimal cleaning:
      - Remove linebreaks/tabs (replace with space)
      - Collapse multiple spaces
      - Strip leading/trailing
      - Preserve punctuation, casing, accented letters
    """
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_normalized(text):
    """
    Fully normalized:
      - Remove linebreaks/tabs
      - Remove all non-ASCII (accented letters are dropped)
      - Convert to lowercase
      - Remove punctuation => keep only [a-z0-9] plus spaces
      - Collapse multiple spaces
      - Strip leading/trailing
    """
    # Remove linebreaks/tabs
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Remove all non-ASCII
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Lowercase
    text = text.lower()

    # Keep only [a-z0-9] + space
    text = re.sub(r"[^a-z0-9 ]+", "", text)

    # Collapse multiple spaces again
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def doc_sort_key(d):
    """
    Parse 'type-XYZ' => integer XYZ => used for sorting.
    Ensures type-10 > type-9 => 'type-9' ... 'type-10' in ascending order.
    """
    prefix, num_str = d.split('-')
    return int(num_str)


def compute_metrics(ref_text, hyp_text, normalized=False):
    """
    Compute Levenshtein distance, CER, WER.
    If normalized=True => use clean_text_normalized,
    else => use clean_text_nonorm.
    """
    if normalized:
        ref_clean = clean_text_normalized(ref_text)
        hyp_clean = clean_text_normalized(hyp_text)
    else:
        ref_clean = clean_text_nonorm(ref_text)
        hyp_clean = clean_text_nonorm(hyp_text)

    dist_char = distance.Levenshtein.distance(ref_clean, hyp_clean)
    ref_len = len(ref_clean)

    cer = dist_char / ref_len if ref_len > 0 else 0.0

    # For WER, split by whitespace
    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    dist_word = distance.Levenshtein.distance("\n".join(ref_words), "\n".join(hyp_words))
    wer = dist_word / len(ref_words) if len(ref_words) > 0 else 0.0

    return dist_char, cer, wer


def process_one_doc(cat, model, doc, ref_text, llm_root, ocr_root, temperature, run):
    """
    - Locate predicted text for doc.
    - If found, return the predicted text, else None.
    """
    if cat == "llm_img2txt":
        pred_path = os.path.join(llm_root, model, doc, f"temperature_{temperature}", run, f"{doc}.txt")
    else:  # OCR
        pred_path = os.path.join(ocr_root, model, doc, run, f"{doc}.txt")

    if not os.path.isfile(pred_path):
        logging.info("  [Missing] No file found for doc '%s' => %s", doc, pred_path)
        return doc, None

    logging.info("  [Found] doc '%s' => %s", doc, pred_path)
    with open(pred_path, "r", encoding="utf-8") as f:
        hyp_text = f.read()

    return doc, hyp_text


def build_html_table(title, doc_names, results_data, doc_lengths, total_doc_len):
    """
    Build an HTML table for a given results_data and doc_lengths structure.
    - results_data[model][doc] => (dist_char, cer, wer)
    - doc_lengths[doc] => length of that doc in the relevant cleaning
    - total_doc_len => sum of all doc lengths in that cleaning
    Returns the HTML string for the table + heading.
    """
    html = f"<h2>{title}</h2>\n"
    html += "<table>\n<tr>\n  <th>Model</th>\n"

    for doc in doc_names:
        html += f"  <th>{doc}</th>\n"
    html += "  <th>Complete Sample</th>\n</tr>\n"

    # Sort model names in alphabetical order
    sorted_models = sorted(results_data.keys(), key=lambda x: x.lower())

    for model in sorted_models:
        html += f'<tr>\n  <td class="model-name">{model}</td>\n'
        for doc in doc_names:
            cell_data = results_data[model].get(doc, None)
            if cell_data is None:
                cell_val = "-"
            else:
                dist_char, cer, wer = cell_data
                doc_len = doc_lengths.get(doc, 0)
                cer_pct = f"{cer*100:.2f}%"
                wer_pct = f"{wer*100:.2f}%"
                # 4 lines:
                cell_val = f"{dist_char}<br>{doc_len}<br>{cer_pct}<br>{wer_pct}"
            html += f"  <td>{cell_val}</td>\n"

        all_data = results_data[model].get("__ALL__", None)
        if all_data is None:
            cell_val = "-"
        else:
            dist_char, cer, wer = all_data
            cer_pct = f"{cer*100:.2f}%"
            wer_pct = f"{wer*100:.2f}%"
            cell_val = f"{dist_char}<br>{total_doc_len}<br>{cer_pct}<br>{wer_pct}"
        html += f"  <td>{cell_val}</td>\n"
        html += "</tr>\n"

    html += "</table>\n"
    return html


def main():
    args = parse_arguments()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    logger.info("Script directory: %s", script_dir)
    logger.info("Project root: %s", project_root)

    # Ground truth
    ground_truth_dir = os.path.join(project_root, "data", "ground_truth", "txt")
    gt_paths = glob.glob(os.path.join(ground_truth_dir, "*.txt"))
    logger.info("Found ground-truth txt files: %s", gt_paths)

    doc_names = [os.path.splitext(os.path.basename(p))[0] for p in gt_paths]
    # Sort doc names numerically
    doc_names.sort(key=doc_sort_key)
    logger.info("Sorted doc_names: %s", doc_names)

    # results/ paths
    llm_root = os.path.join(project_root, "results", "llm_img2txt")
    ocr_root = os.path.join(project_root, "results", "ocr_img2txt")

    # Models
    llm_models = []
    if os.path.isdir(llm_root):
        llm_models = [m for m in os.listdir(llm_root) if os.path.isdir(os.path.join(llm_root, m))]
    ocr_models = []
    if os.path.isdir(ocr_root):
        ocr_models = [m for m in os.listdir(ocr_root) if os.path.isdir(os.path.join(ocr_root, m))]

    all_models = [("llm_img2txt", m) for m in llm_models] + [("ocr_img2txt", m) for m in ocr_models]
    # sort by model name
    all_models.sort(key=lambda x: x[1].lower())

    # Load ground truth + compute doc lengths for both normalized & non-normalized
    ground_truths = {}
    doc_lengths_normalized = {}
    doc_lengths_nonorm = {}

    for doc in doc_names:
        path = os.path.join(ground_truth_dir, f"{doc}.txt")
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        ground_truths[doc] = txt
        # Normalized length
        doc_lengths_normalized[doc] = len(clean_text_normalized(txt))
        # Non-normalized minimal length
        doc_lengths_nonorm[doc] = len(clean_text_nonorm(txt))

    # Prepare data structures for both normalized & non-normalized
    results_data_normalized = {}
    results_data_nonorm = {}
    aggregated_text_normalized = {}
    aggregated_text_nonorm = {}

    for cat, model in all_models:
        results_data_normalized[model] = {}
        results_data_nonorm[model] = {}
        aggregated_text_normalized[model] = {"ref": [], "hyp": []}
        aggregated_text_nonorm[model] = {"ref": [], "hyp": []}

    # Parallel process each model => gather predicted text
    for cat, model in all_models:
        logger.info("Processing model: %s (category: %s)", model, cat)

        tasks = []
        for doc in doc_names:
            ref_txt = ground_truths[doc]
            tasks.append((cat, model, doc, ref_txt, llm_root, ocr_root, args.temperature, args.run))

        parallel_results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_one_doc)(*t) for t in tasks
        )

        # For each doc, compute both normalized and non-normalized metrics
        for doc, hyp_txt in parallel_results:
            if hyp_txt is None:
                # Missing prediction
                results_data_normalized[model][doc] = None
                results_data_nonorm[model][doc] = None
            else:
                # Compute normalized
                dist_char_n, cer_n, wer_n = compute_metrics(ground_truths[doc], hyp_txt, normalized=True)
                results_data_normalized[model][doc] = (dist_char_n, cer_n, wer_n)
                aggregated_text_normalized[model]["ref"].append(ground_truths[doc])
                aggregated_text_normalized[model]["hyp"].append(hyp_txt)

                # Compute non-normalized
                dist_char_nn, cer_nn, wer_nn = compute_metrics(ground_truths[doc], hyp_txt, normalized=False)
                results_data_nonorm[model][doc] = (dist_char_nn, cer_nn, wer_nn)
                aggregated_text_nonorm[model]["ref"].append(ground_truths[doc])
                aggregated_text_nonorm[model]["hyp"].append(hyp_txt)

    # Compute "Complete Sample" for both normalized & non-normalized
    for cat, model in all_models:
        ref_concat_normalized = "\n".join(aggregated_text_normalized[model]["ref"]).strip()
        hyp_concat_normalized = "\n".join(aggregated_text_normalized[model]["hyp"]).strip()

        if len(ref_concat_normalized) == 0 or len(hyp_concat_normalized) == 0:
            results_data_normalized[model]["__ALL__"] = None
        else:
            dist_char, cer, wer = compute_metrics(ref_concat_normalized, hyp_concat_normalized, normalized=True)
            results_data_normalized[model]["__ALL__"] = (dist_char, cer, wer)

        ref_concat_nonorm = "\n".join(aggregated_text_nonorm[model]["ref"]).strip()
        hyp_concat_nonorm = "\n".join(aggregated_text_nonorm[model]["hyp"]).strip()

        if len(ref_concat_nonorm) == 0 or len(hyp_concat_nonorm) == 0:
            results_data_nonorm[model]["__ALL__"] = None
        else:
            dist_char, cer, wer = compute_metrics(ref_concat_nonorm, hyp_concat_nonorm, normalized=False)
            results_data_nonorm[model]["__ALL__"] = (dist_char, cer, wer)

    # total doc length across all docs (normalized & non-normalized)
    all_ref_concat_normalized = "\n".join(clean_text_normalized(ground_truths[d]) for d in doc_names)
    total_doc_len_normalized = len(all_ref_concat_normalized)
    all_ref_concat_nonorm = "\n".join(clean_text_nonorm(ground_truths[d]) for d in doc_names)
    total_doc_len_nonorm = len(all_ref_concat_nonorm)

    # ---------------- BUILD HTML ----------------
    logger.info("Building HTML...")

    html_head = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>OCR vs. LLM Benchmarking Results</title>
<style>
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  margin: 40px;
  background: #fdfdfd;
  color: #333;
}
h1 {
  text-align: center;
  font-size: 2em;
  margin-bottom: 0.25em;
}
h2 {
  margin-top: 1.5em;
  margin-bottom: 0.5em;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 2em;
  background: #fff;
}
th {
  background: #4CAF50;
  color: white;
  text-align: center;
  padding: 8px;
  border: 1px solid #ddd;
}
td {
  border: 1px solid #ddd;
  padding: 8px;
  vertical-align: middle;
  text-align: center;
  line-height: 1.5em;
}
tr:nth-child(even) {
  background: #f2f2f2;
}
.model-name {
  font-weight: bold;
  background: #f9f9de !important;
}
.note {
  font-size: 0.95em;
  line-height: 1.4em;
  color: #555;
  border-top: 2px solid #ccc;
  padding-top: 1em;
  margin-top: 2em;
}
ul {
  margin: 0;
  padding-left: 1.2em;
}
</style>
</head>
<body>
<h1>OCR vs. LLM Benchmarking Results (Parallel + RapidFuzz)</h1>
"""

    # Build the two tables
    table_normalized = build_html_table(
        "Normalized Results (ASCII-only, punctuation removed, lowercased)",
        doc_names, results_data_normalized,
        doc_lengths_normalized,
        total_doc_len_normalized
    )

    table_nonorm = build_html_table(
        "Non-normalized Results (punctuation & casing preserved, minimal cleaning)",
        doc_names, results_data_nonorm,
        doc_lengths_nonorm,
        total_doc_len_nonorm
    )

    # Combine the tables + a final note
    used_cores = (args.n_jobs if args.n_jobs != -1 else os.cpu_count())
    note = f"""<div class="note">
<strong>Notes:</strong>
<ul>
  <li>Documents are sorted numerically: type-1, type-2, ..., type-9, type-10, etc.</li>
  <li>Each cell displays 4 lines: 
    <br>&emsp;1) Levenshtein distance 
    <br>&emsp;2) Ground-truth document length 
    <br>&emsp;3) CER% 
    <br>&emsp;4) WER%</li>
  <li>Two tables are shown: 
    <br>&emsp;- Normalized: ASCII-only, punctuation removed, lowercased, etc.
    <br>&emsp;- Non-normalized: punctuation/accent/case preserved, linebreaks removed, etc.</li>
  <li>Parallelized with joblib, using n_jobs={used_cores}.</li>
  <li>Levenshtein distance is computed via <em>rapidfuzz</em> (C++ backend for speed).</li>
  <li>CER (Character Error Rate) = (distance / reference_length) * 100%, within the given table's cleaning.</li>
  <li>WER (Word Error Rate) uses a word-level edit distance. Tokenization depends on the cleaning version.</li>
  <li>Missing predictions are marked with "-".</li>
</ul>
</div>
</body>
</html>"""

    full_html = html_head + table_normalized + table_nonorm + note
    output_path = os.path.join(script_dir, args.output_html)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    logger.info("All done! Results saved to '%s'. Enjoy your world-class benchmarking tables!", output_path)


if __name__ == "__main__":
    main()