#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking text extraction accuracy for the *revisions* experiment.

Adapted from src/benchmarking/txt_accuracy.py.  Key differences:
  - LLM predictions are read from  results_revisions/llm_img2txt/  (not results/).
  - OCR baselines (Transkribus, pytesseract) are still read from the OLD
    results/ocr_img2txt/  directory so they can be included as reference rows.
  - Produces LaTeX .tex table files (in addition to the HTML report).

Generates TWO result tables (each in both normalized and non-normalized form):

  1) Normalized Results Table
     - Non-ASCII characters removed entirely.
     - Lower-cased.
     - Punctuation removed (only [a-z0-9] and spaces survive).
     - Whitespace collapsed and stripped.

  2) Non-normalized Results Table
     - Punctuation, casing, and accented characters preserved.
     - Line breaks / tabs replaced with spaces.
     - Whitespace collapsed and stripped.

Each cell in both tables contains four metrics:
     1) Levenshtein distance  (character-level edit distance)
     2) Ground-truth document length  (in the relevant cleaning mode)
     3) CER%  (Character Error Rate = distance / gt_length * 100)
     4) WER%  (Word Error Rate, word-level edit distance / word count * 100)

A "Complete Sample" column concatenates all documents for an aggregate metric.

Usage example:
  python txt_accuracy.py --temperature 0.0 --run run_01
"""

import os
import re
import glob
import argparse
import logging
from joblib import Parallel, delayed
from rapidfuzz import distance

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


##############################################################################
# CLI argument parsing
##############################################################################

def parse_arguments():
    """Parse command-line arguments for the text benchmarking script."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute text extraction metrics (Levenshtein, CER, WER) "
            "for the revisions experiment, parallelized with joblib + rapidfuzz."
        )
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0.0",
        help="Temperature folder name for LLM results (default: 0.0).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="run_01",
        help="Run folder name (default: run_01).",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="txt_accuracy_revisions.html",
        help="Name of the output HTML report file (default: txt_accuracy_revisions.html).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel joblib workers (-1 = all cores, default: -1).",
    )
    return parser.parse_args()


##############################################################################
# Text cleaning functions
##############################################################################

def clean_text_nonorm(text):
    """
    Minimal cleaning that preserves content fidelity:
      - Replace line breaks and tabs with spaces.
      - Collapse multiple whitespace characters into a single space.
      - Strip leading and trailing whitespace.
      - Punctuation, casing, and accented characters are PRESERVED.
    """
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_normalized(text):
    """
    Aggressive normalization for fair cross-model comparison:
      1. Replace line breaks and tabs with spaces.
      2. Collapse whitespace and strip.
      3. Remove ALL non-ASCII characters (accented letters are dropped).
      4. Convert to lower case.
      5. Remove everything except [a-z0-9] and spaces.
      6. Collapse whitespace again and strip.
    """
    # Step 1-2: remove line breaks / tabs, collapse whitespace.
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r"\s+", " ", text).strip()
    # Step 3: drop non-ASCII bytes entirely.
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Step 4: lower-case.
    text = text.lower()
    # Step 5: keep only alphanumeric + space.
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    # Step 6: collapse whitespace one more time.
    text = re.sub(r"\s+", " ", text).strip()
    return text


##############################################################################
# Sorting helper
##############################################################################

def doc_sort_key(d):
    """
    Parse a document name like 'type-10' into an integer (10) for numeric
    sorting.  This ensures type-9 sorts before type-10.
    """
    _prefix, num_str = d.split("-")
    return int(num_str)


##############################################################################
# Metric computation
##############################################################################

def compute_metrics(ref_text, hyp_text, normalized=False):
    """
    Compute three metrics between a reference and hypothesis text:
      1. Levenshtein distance (character-level)
      2. CER  (Character Error Rate = distance / reference_length)
      3. WER  (Word Error Rate  = word-level edit distance / reference word count)

    If normalized=True, both texts are cleaned with clean_text_normalized();
    otherwise clean_text_nonorm() is used.
    """
    if normalized:
        ref_clean = clean_text_normalized(ref_text)
        hyp_clean = clean_text_normalized(hyp_text)
    else:
        ref_clean = clean_text_nonorm(ref_text)
        hyp_clean = clean_text_nonorm(hyp_text)

    # Character-level Levenshtein distance (via rapidfuzz C++ backend).
    dist_char = distance.Levenshtein.distance(ref_clean, hyp_clean)
    ref_len = len(ref_clean)

    # CER: character error rate.
    cer = dist_char / ref_len if ref_len > 0 else 0.0

    # WER: word error rate (tokenise by whitespace, then word-level Levenshtein).
    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    # Join words with newlines so Levenshtein operates on word tokens.
    dist_word = distance.Levenshtein.distance(
        "\n".join(ref_words), "\n".join(hyp_words)
    )
    wer = dist_word / len(ref_words) if len(ref_words) > 0 else 0.0

    return dist_char, cer, wer


##############################################################################
# Parallel per-document worker
##############################################################################

def resolve_temperature(base_dir, temperature):
    """Return the temperature that has a matching folder.

    Tries the requested temperature first; falls back to 0.0 then 1.0.
    This is needed because GPT-5.5 (a reasoning model) ignores the
    temperature parameter and only has results under temperature_1.0/.
    """
    if os.path.isdir(os.path.join(base_dir, f"temperature_{temperature}")):
        return temperature
    for fallback in ["0.0", "1.0"]:
        if os.path.isdir(os.path.join(base_dir, f"temperature_{fallback}")):
            return fallback
    return temperature


def process_one_doc(cat, model, doc, ref_text, llm_root, ocr_root, temperature, run):
    """
    Locate and read the predicted text file for one (model, document) pair.

    LLM predictions follow the path:
      {llm_root}/{model}/{doc}/temperature_{T}/{run}/{doc}.txt

    OCR predictions follow the path:
      {ocr_root}/{model}/{doc}/{run}/{doc}.txt

    Returns (doc, predicted_text_or_None).
    """
    if cat == "llm_img2txt":
        # LLM-based predictions live under results_revisions/.
        model_base = os.path.join(llm_root, model, doc)
        temp = resolve_temperature(model_base, temperature)
        pred_path = os.path.join(
            model_base,
            f"temperature_{temp}", run, f"{doc}.txt",
        )
    else:
        # OCR baselines live under the OLD results/ directory.
        pred_path = os.path.join(
            ocr_root, model, doc, run, f"{doc}.txt",
        )

    if not os.path.isfile(pred_path):
        logging.info("  [Missing] No file found for doc '%s' => %s", doc, pred_path)
        return doc, None

    logging.info("  [Found] doc '%s' => %s", doc, pred_path)
    with open(pred_path, "r", encoding="utf-8") as f:
        hyp_text = f.read()

    return doc, hyp_text


##############################################################################
# HTML table builder
##############################################################################

def build_html_table(title, doc_names, results_data, doc_lengths, total_doc_len):
    """
    Build an HTML table string for one cleaning mode (normalized or non-norm).

    Parameters:
      title          -- heading text above the table.
      doc_names      -- ordered list of document names (type-1 .. type-10).
      results_data   -- dict[model][doc] -> (dist_char, cer, wer) or None.
      doc_lengths    -- dict[doc] -> int (ground-truth length in this cleaning).
      total_doc_len  -- int, total GT length across all docs.

    Each cell displays four lines:
      1) Levenshtein distance
      2) GT document length
      3) CER%
      4) WER%
    """
    html = f"<h2>{title}</h2>\n"
    html += "<table>\n<tr>\n  <th>Model</th>\n"

    for doc in doc_names:
        html += f"  <th>{doc}</th>\n"
    html += "  <th>Complete Sample</th>\n</tr>\n"

    # Sort model names alphabetically (case-insensitive).
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
                cer_pct = f"{cer * 100:.2f}%"
                wer_pct = f"{wer * 100:.2f}%"
                # Four lines per cell: distance, GT length, CER%, WER%.
                cell_val = f"{dist_char}<br>{doc_len}<br>{cer_pct}<br>{wer_pct}"
            html += f"  <td>{cell_val}</td>\n"

        # "Complete Sample" column: aggregate across all documents.
        all_data = results_data[model].get("__ALL__", None)
        if all_data is None:
            cell_val = "-"
        else:
            dist_char, cer, wer = all_data
            cer_pct = f"{cer * 100:.2f}%"
            wer_pct = f"{wer * 100:.2f}%"
            cell_val = f"{dist_char}<br>{total_doc_len}<br>{cer_pct}<br>{wer_pct}"
        html += f"  <td>{cell_val}</td>\n"
        html += "</tr>\n"

    html += "</table>\n"
    return html


##############################################################################
# LaTeX table builder
##############################################################################

def _escape_latex(text: str) -> str:
    """Escape characters that have special meaning in LaTeX."""
    text = text.replace("_", r"\_")
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("#", r"\#")
    return text


def build_latex_table(title, doc_names, results_data, doc_lengths, total_doc_len,
                      table_number, metric="CER"):
    """
    Build a LaTeX table showing one metric (CER or WER) per cell for one
    cleaning mode.

    The table has one row per model and one column per document plus a
    "Complete Sample" column.

    Parameters:
      title          -- caption text for the table.
      doc_names      -- ordered list of document names.
      results_data   -- dict[model][doc] -> (dist_char, cer, wer) or None.
      doc_lengths    -- dict[doc] -> int (GT length for this cleaning).
      total_doc_len  -- int, total GT length.
      table_number   -- integer used in \\label{tab:...}.
      metric         -- "CER" or "WER" (selects which value to display).
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        rf"\caption{{{_escape_latex(title)}."
        r" Gemini-3.1 uses temperature 0.0."
        r" GPT-5.5 uses reasoning\_effort=high (temperature 1.0).}"
    )
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column spec: l for model name + one c per document + one c for Complete Sample.
    n_cols = len(doc_names) + 1  # +1 for "Complete Sample"
    col_spec = "l|" + "c" * n_cols
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")

    # Header row.
    header_parts = ["Model"]
    for doc in doc_names:
        header_parts.append(_escape_latex(doc))
    header_parts.append("Complete Sample")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\hline")

    # Determine which index to pull from the (dist, cer, wer) tuple.
    # 0 = Levenshtein distance, 1 = CER, 2 = WER.
    metric_idx = 1 if metric == "CER" else 2

    # One row per model.
    sorted_models = sorted(results_data.keys(), key=lambda x: x.lower())
    for model in sorted_models:
        row_parts = [_escape_latex(model)]
        for doc in doc_names:
            cell_data = results_data[model].get(doc, None)
            if cell_data is None:
                row_parts.append("-")
            else:
                val = cell_data[metric_idx] * 100.0
                row_parts.append(f"{val:.2f}\\%")

        # Complete Sample column.
        all_data = results_data[model].get("__ALL__", None)
        if all_data is None:
            row_parts.append("-")
        else:
            val = all_data[metric_idx] * 100.0
            row_parts.append(f"{val:.2f}\\%")

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\hline")

    # Footer row: GT doc lengths.
    len_parts = ["GT length"]
    for doc in doc_names:
        len_parts.append(str(doc_lengths.get(doc, 0)))
    len_parts.append(str(total_doc_len))
    lines.append(" & ".join(len_parts) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("}")  # close resizebox
    lines.append(rf"\label{{tab:revision_txt_table{table_number}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


##############################################################################
# Main entry point
##############################################################################

def main():
    args = parse_arguments()

    # Resolve project paths relative to this script's location.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    logger.info("Script directory: %s", script_dir)
    logger.info("Project root: %s", project_root)

    # Directory for LaTeX .tex output files.
    tables_dir = os.path.join(script_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load ground-truth text files from data/ground_truth/txt/.
    # ------------------------------------------------------------------
    ground_truth_dir = os.path.join(project_root, "data", "ground_truth", "txt")
    gt_paths = glob.glob(os.path.join(ground_truth_dir, "*.txt"))
    logger.info("Found ground-truth txt files: %s", gt_paths)

    # Extract document names and sort numerically (type-1 < type-2 < ... < type-10).
    doc_names = [os.path.splitext(os.path.basename(p))[0] for p in gt_paths]
    doc_names.sort(key=doc_sort_key)
    logger.info("Sorted doc_names: %s", doc_names)

    # ------------------------------------------------------------------
    # 2) Set up paths for LLM predictions and OCR baselines.
    # ------------------------------------------------------------------

    # LLM predictions come from the REVISIONS directory.
    llm_root = os.path.join(project_root, "results_revisions", "llm_img2txt")

    # OCR baselines come from the ORIGINAL results/ directory (read-only reference).
    ocr_root = os.path.join(project_root, "results", "ocr_img2txt")

    # ------------------------------------------------------------------
    # 3) Discover available models.
    # ------------------------------------------------------------------
    llm_models = []
    if os.path.isdir(llm_root):
        llm_models = [
            m for m in os.listdir(llm_root)
            if os.path.isdir(os.path.join(llm_root, m))
        ]

    ocr_models = []
    if os.path.isdir(ocr_root):
        ocr_models = [
            m for m in os.listdir(ocr_root)
            if os.path.isdir(os.path.join(ocr_root, m))
        ]

    # Tag each model with its category so we know where to look for files.
    all_models = (
        [("llm_img2txt", m) for m in llm_models]
        + [("ocr_img2txt", m) for m in ocr_models]
    )
    # Sort alphabetically by model name for deterministic output.
    all_models.sort(key=lambda x: x[1].lower())

    logger.info(
        "Discovered %d model(s): %s",
        len(all_models),
        [(c, m) for c, m in all_models],
    )

    # ------------------------------------------------------------------
    # 4) Load ground-truth texts and pre-compute document lengths for
    #    both cleaning modes.
    # ------------------------------------------------------------------
    ground_truths = {}               # doc -> raw text
    doc_lengths_normalized = {}      # doc -> len(clean_text_normalized(text))
    doc_lengths_nonorm = {}          # doc -> len(clean_text_nonorm(text))

    for doc in doc_names:
        path = os.path.join(ground_truth_dir, f"{doc}.txt")
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        ground_truths[doc] = txt
        doc_lengths_normalized[doc] = len(clean_text_normalized(txt))
        doc_lengths_nonorm[doc] = len(clean_text_nonorm(txt))

    # ------------------------------------------------------------------
    # 5) Prepare per-model result containers.
    # ------------------------------------------------------------------
    # Each dict maps  model_name -> { doc -> (dist, cer, wer) | None }.
    results_data_normalized = {}
    results_data_nonorm = {}
    # Aggregation buffers: raw ref/hyp texts to concatenate for "Complete Sample".
    aggregated_text_normalized = {}
    aggregated_text_nonorm = {}

    for cat, model in all_models:
        results_data_normalized[model] = {}
        results_data_nonorm[model] = {}
        aggregated_text_normalized[model] = {"ref": [], "hyp": []}
        aggregated_text_nonorm[model] = {"ref": [], "hyp": []}

    # ------------------------------------------------------------------
    # 6) For each model, process all documents in parallel.
    # ------------------------------------------------------------------
    for cat, model in all_models:
        logger.info("Processing model: %s (category: %s)", model, cat)

        # Build the task list for joblib.
        tasks = []
        for doc in doc_names:
            ref_txt = ground_truths[doc]
            tasks.append(
                (cat, model, doc, ref_txt, llm_root, ocr_root, args.temperature, args.run)
            )

        # Execute in parallel: each task reads one predicted text file.
        parallel_results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_one_doc)(*t) for t in tasks
        )

        # Compute metrics for each document in both cleaning modes.
        for doc, hyp_txt in parallel_results:
            if hyp_txt is None:
                # Prediction file was missing for this document.
                results_data_normalized[model][doc] = None
                results_data_nonorm[model][doc] = None
            else:
                # --- Normalized metrics ---
                dist_n, cer_n, wer_n = compute_metrics(
                    ground_truths[doc], hyp_txt, normalized=True
                )
                results_data_normalized[model][doc] = (dist_n, cer_n, wer_n)
                aggregated_text_normalized[model]["ref"].append(ground_truths[doc])
                aggregated_text_normalized[model]["hyp"].append(hyp_txt)

                # --- Non-normalized metrics ---
                dist_nn, cer_nn, wer_nn = compute_metrics(
                    ground_truths[doc], hyp_txt, normalized=False
                )
                results_data_nonorm[model][doc] = (dist_nn, cer_nn, wer_nn)
                aggregated_text_nonorm[model]["ref"].append(ground_truths[doc])
                aggregated_text_nonorm[model]["hyp"].append(hyp_txt)

    # ------------------------------------------------------------------
    # 7) Compute the "Complete Sample" aggregate for each model.
    # ------------------------------------------------------------------
    for cat, model in all_models:
        # Normalized aggregate: concatenate all ref/hyp texts, then compute.
        ref_concat_norm = "\n".join(aggregated_text_normalized[model]["ref"]).strip()
        hyp_concat_norm = "\n".join(aggregated_text_normalized[model]["hyp"]).strip()

        if len(ref_concat_norm) == 0 or len(hyp_concat_norm) == 0:
            results_data_normalized[model]["__ALL__"] = None
        else:
            dist_char, cer, wer = compute_metrics(
                ref_concat_norm, hyp_concat_norm, normalized=True
            )
            results_data_normalized[model]["__ALL__"] = (dist_char, cer, wer)

        # Non-normalized aggregate.
        ref_concat_nn = "\n".join(aggregated_text_nonorm[model]["ref"]).strip()
        hyp_concat_nn = "\n".join(aggregated_text_nonorm[model]["hyp"]).strip()

        if len(ref_concat_nn) == 0 or len(hyp_concat_nn) == 0:
            results_data_nonorm[model]["__ALL__"] = None
        else:
            dist_char, cer, wer = compute_metrics(
                ref_concat_nn, hyp_concat_nn, normalized=False
            )
            results_data_nonorm[model]["__ALL__"] = (dist_char, cer, wer)

    # ------------------------------------------------------------------
    # 8) Compute total GT length across all documents for each mode.
    # ------------------------------------------------------------------
    # We concatenate cleaned GT strings (separated by newlines) to get the
    # total length that corresponds to the "Complete Sample" column.
    all_ref_concat_normalized = "\n".join(
        clean_text_normalized(ground_truths[d]) for d in doc_names
    )
    total_doc_len_normalized = len(all_ref_concat_normalized)

    all_ref_concat_nonorm = "\n".join(
        clean_text_nonorm(ground_truths[d]) for d in doc_names
    )
    total_doc_len_nonorm = len(all_ref_concat_nonorm)

    # ------------------------------------------------------------------
    # 9) Generate LaTeX tables.
    # ------------------------------------------------------------------
    logger.info("Generating LaTeX tables in %s ...", tables_dir)

    # We produce 4 LaTeX tables:
    #   Table 1: Normalized CER
    #   Table 2: Normalized WER
    #   Table 3: Non-normalized CER
    #   Table 4: Non-normalized WER
    latex_specs = [
        (1, "CER", "Normalized CER (ASCII-only, punctuation removed, lowercased)",
         results_data_normalized, doc_lengths_normalized, total_doc_len_normalized),
        (2, "WER", "Normalized WER (ASCII-only, punctuation removed, lowercased)",
         results_data_normalized, doc_lengths_normalized, total_doc_len_normalized),
        (3, "CER", "Non-normalized CER (punctuation and casing preserved)",
         results_data_nonorm, doc_lengths_nonorm, total_doc_len_nonorm),
        (4, "WER", "Non-normalized WER (punctuation and casing preserved)",
         results_data_nonorm, doc_lengths_nonorm, total_doc_len_nonorm),
    ]

    for tbl_num, metric, caption, rdata, dlens, total_len in latex_specs:
        latex_str = build_latex_table(
            title=caption,
            doc_names=doc_names,
            results_data=rdata,
            doc_lengths=dlens,
            total_doc_len=total_len,
            table_number=tbl_num,
            metric=metric,
        )
        tex_filename = f"txt_table{tbl_num}_{metric.lower()}.tex"
        tex_path = os.path.join(tables_dir, tex_filename)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        logger.info("  Wrote %s", tex_path)

    # ------------------------------------------------------------------
    # 10) Generate the HTML report.
    # ------------------------------------------------------------------
    logger.info("Building HTML report...")

    html_head = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Text Extraction Benchmarking (Revisions)</title>
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
<h1>Text Extraction Benchmark &mdash; Revisions (Parallel + RapidFuzz)</h1>
"""

    # Build the two HTML tables (normalized and non-normalized).
    table_normalized = build_html_table(
        "Normalized Results (ASCII-only, punctuation removed, lowercased)",
        doc_names,
        results_data_normalized,
        doc_lengths_normalized,
        total_doc_len_normalized,
    )

    table_nonorm = build_html_table(
        "Non-normalized Results (punctuation & casing preserved, minimal cleaning)",
        doc_names,
        results_data_nonorm,
        doc_lengths_nonorm,
        total_doc_len_nonorm,
    )

    # Footer with methodology notes.
    used_cores = args.n_jobs if args.n_jobs != -1 else os.cpu_count()
    note = f"""<div class="note">
<strong>Notes:</strong>
<ul>
  <li>Documents are sorted numerically: type-1, type-2, ..., type-9, type-10.</li>
  <li>Each cell displays 4 lines:
    <br>&emsp;1) Levenshtein distance
    <br>&emsp;2) Ground-truth document length
    <br>&emsp;3) CER%
    <br>&emsp;4) WER%</li>
  <li>Two tables are shown:
    <br>&emsp;- Normalized: ASCII-only, punctuation removed, lowercased.
    <br>&emsp;- Non-normalized: punctuation/accent/case preserved, linebreaks removed.</li>
  <li>LLM predictions read from <code>results_revisions/llm_img2txt/</code>.</li>
  <li>OCR baselines read from <code>results/ocr_img2txt/</code> (original experiment).</li>
  <li>Parallelized with joblib, using n_jobs={used_cores}.</li>
  <li>Levenshtein distance computed via <em>rapidfuzz</em> (C++ backend).</li>
  <li>CER (Character Error Rate) = (distance / reference_length) * 100%.</li>
  <li>WER (Word Error Rate) uses word-level edit distance / reference word count.</li>
  <li>Missing predictions are marked with "-".</li>
  <li>LaTeX tables saved to <code>{tables_dir}</code>.</li>
</ul>
</div>
</body>
</html>"""

    full_html = html_head + table_normalized + table_nonorm + note
    output_path = os.path.join(script_dir, args.output_html)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    logger.info(
        "Done!  HTML report saved to '%s'.  LaTeX tables in '%s'.",
        output_path,
        tables_dir,
    )


if __name__ == "__main__":
    main()
