#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX tables for the revision experiments, formatted to match the
paper's existing tables (Tables 3, 4, 5 in the draft).

Produces a single file: src/benchmarking_revisions/tables/revision_tables.tex

Contains:
  - Table A: OCR & Post-OCR results (CER + WER, normalized & non-normalized)
             for Gemini 3.1 Pro Preview and GPT-5.5
  - Table B: NER results (Strict + Fuzzy match) by document type
             for Gemini 3.1 and GPT-5.5 (GT TXT-to-CSV and PNG-to-CSV)
  - Table C: NER results by variable (aggregated)

The city-year names match the paper convention:
  type-1=Aachen-1838, type-2=Dresden-1797, type-3=Leipzig-1753,
  type-4=Frankfurt-1860, type-5=Frankfurt-1778, type-6=Lübeck-1870,
  type-7=Dresden-1819, type-8=Riga-1810, type-9=Leipzig-1800,
  type-10=Trier-1853
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import distance
from rapidfuzz.distance import JaroWinkler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

###############################################################################
# Configuration
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Map type-N to city-year labels used in the paper
DOC_LABELS = {
    "type-1": "Aachen-1838",
    "type-2": "Dresden-1797",
    "type-3": "Leipzig-1753",
    "type-4": "Frankfurt-1860",
    "type-5": "Frankfurt-1778",
    "type-6": r"L\"{u}beck-1870",
    "type-7": "Dresden-1819",
    "type-8": "Riga-1810",
    "type-9": "Leipzig-1800",
    "type-10": "Trier-1853",
}
DOC_NAMES = [f"type-{i}" for i in range(1, 11)]

# CSV columns used for NER comparison (excluding "id")
EXPECTED_COLUMNS = ["first and middle names", "surname", "occupation", "address"]
FUZZY_THRESHOLD = 0.90

###############################################################################
# Text cleaning (identical to benchmarking scripts)
###############################################################################

def clean_text_nonorm(text):
    """Minimal cleaning: replace whitespace chars, collapse, strip."""
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_normalized(text):
    """Aggressive: ASCII-only, lowercase, alphanumeric+space only."""
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

###############################################################################
# CSV normalization (identical to benchmarking scripts)
###############################################################################

def normalize_cell(x):
    if x is None: return ""
    x = str(x).strip().lower()
    if x in ("", "null"): return ""
    return x

def load_csv(path):
    if not os.path.isfile(path): return None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
        for col in EXPECTED_COLUMNS:
            if col not in df.columns: return None
        df = df.reindex(columns=EXPECTED_COLUMNS)
        for col in df.columns:
            df[col] = df[col].apply(normalize_cell)
        return df
    except Exception:
        return None

###############################################################################
# Metric helpers
###############################################################################

def compute_txt_metrics(ref_text, hyp_text, normalized=False):
    """Return (cer, wer) as percentages."""
    if normalized:
        ref_clean = clean_text_normalized(ref_text)
        hyp_clean = clean_text_normalized(hyp_text)
    else:
        ref_clean = clean_text_nonorm(ref_text)
        hyp_clean = clean_text_nonorm(hyp_text)

    dist_char = distance.Levenshtein.distance(ref_clean, hyp_clean)
    ref_len = len(ref_clean)
    cer = (dist_char / ref_len * 100) if ref_len > 0 else 0.0

    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    dist_word = distance.Levenshtein.distance("\n".join(ref_words), "\n".join(hyp_words))
    wer = (dist_word / len(ref_words) * 100) if len(ref_words) > 0 else 0.0

    return cer, wer

def compute_csv_metrics(gt_df, pred_df):
    """Return (strict_pct, fuzzy_pct, per_col_strict, per_col_fuzzy) or None if dimension mismatch."""
    if gt_df is None or pred_df is None:
        return None
    if gt_df.shape[0] != pred_df.shape[0]:
        return None

    n_rows = gt_df.shape[0]
    n_cols = len(EXPECTED_COLUMNS)
    total = n_rows * n_cols

    strict_matches = (gt_df.values == pred_df.values).sum()
    strict_pct = strict_matches / total * 100 if total > 0 else 0.0

    fuzzy_matches = 0
    for r in range(n_rows):
        for c in range(n_cols):
            sim = JaroWinkler.similarity(
                gt_df.iat[r, c].strip().lower(),
                pred_df.iat[r, c].strip().lower()
            )
            if sim >= FUZZY_THRESHOLD:
                fuzzy_matches += 1
    fuzzy_pct = fuzzy_matches / total * 100 if total > 0 else 0.0

    # Per-column metrics
    per_col_strict = {}
    per_col_fuzzy = {}
    for ci, col in enumerate(EXPECTED_COLUMNS):
        col_strict = sum(1 for r in range(n_rows) if gt_df.iat[r, ci] == pred_df.iat[r, ci])
        col_fuzzy = sum(1 for r in range(n_rows)
                        if JaroWinkler.similarity(gt_df.iat[r, ci].strip().lower(),
                                                   pred_df.iat[r, ci].strip().lower()) >= FUZZY_THRESHOLD)
        per_col_strict[col] = col_strict / n_rows * 100 if n_rows > 0 else 0.0
        per_col_fuzzy[col] = col_fuzzy / n_rows * 100 if n_rows > 0 else 0.0

    return strict_pct, fuzzy_pct, per_col_strict, per_col_fuzzy

###############################################################################
# LaTeX helpers
###############################################################################

def esc(s):
    """Escape LaTeX special characters."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")

def fmt_pct(v):
    """Format a percentage value for LaTeX."""
    return f"{v:.2f}\\%"

###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Generate revision LaTeX tables")
    parser.add_argument("--temperature", default="0.0")
    parser.add_argument("--run", default="run_01")
    args = parser.parse_args()

    results_rev = PROJECT_ROOT / "results_revisions"
    results_old = PROJECT_ROOT / "results"
    gt_txt_dir = PROJECT_ROOT / "data" / "ground_truth" / "txt"
    gt_csv_dir = PROJECT_ROOT / "data" / "ground_truth" / "csv"
    out_dir = PROJECT_ROOT / "src" / "benchmarking_revisions" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("% Auto-generated revision tables")
    lines.append("% Gemini 3.1 Pro Preview (temperature 0.0, thinking_budget=-1 dynamic)")
    lines.append("% GPT-5.5 (reasoning_effort=high, temperature=default 1.0)")
    lines.append("")

    # =========================================================================
    # TABLE A: OCR & Post-OCR (CER + WER)
    # =========================================================================
    logger.info("Computing OCR metrics...")

    # Models to include in OCR table
    ocr_models = [
        ("llm_img2txt", "gemini-3.1", r"\textbf{Gemini 3.1 Pro}"),
        ("llm_img2txt", "gpt-5.5", r"\textbf{GPT-5.5}"),
        ("llm_img2txt", "gemini-3.1-with-transkribus",
         r"\makecell[c]{\textbf{Transkribus}\\\textbf{Print M1}\\$\downarrow$\\\textbf{Gemini 3.1 Pro}}"),
        ("llm_img2txt", "gpt-5.5-with-transkribus",
         r"\makecell[c]{\textbf{Transkribus}\\\textbf{Print M1}\\$\downarrow$\\\textbf{GPT-5.5}}"),
    ]

    # Collect metrics: {(model, doc, normalized): (cer, wer)}
    ocr_data = {}
    for cat, model, _label in ocr_models:
        for doc in DOC_NAMES:
            # Load ground truth
            gt_path = gt_txt_dir / f"{doc}.txt"
            if not gt_path.is_file():
                continue
            gt_text = gt_path.read_text(encoding="utf-8")

            # Load prediction
            if cat == "llm_img2txt":
                pred_path = results_rev / cat / model / doc / f"temperature_{args.temperature}" / args.run / f"{doc}.txt"
            else:
                pred_path = results_old / cat / model / doc / args.run / f"{doc}.txt"

            if not pred_path.is_file():
                ocr_data[(model, doc, True)] = None
                ocr_data[(model, doc, False)] = None
                continue

            hyp_text = pred_path.read_text(encoding="utf-8")
            cer_n, wer_n = compute_txt_metrics(gt_text, hyp_text, normalized=True)
            cer_nn, wer_nn = compute_txt_metrics(gt_text, hyp_text, normalized=False)
            ocr_data[(model, doc, True)] = (cer_n, wer_n)
            ocr_data[(model, doc, False)] = (cer_nn, wer_nn)

    # Compute full sample (concatenated)
    for cat, model, _label in ocr_models:
        for norm in [True, False]:
            all_gt = []
            all_hyp = []
            for doc in DOC_NAMES:
                gt_path = gt_txt_dir / f"{doc}.txt"
                if not gt_path.is_file():
                    continue
                if cat == "llm_img2txt":
                    pred_path = results_rev / cat / model / doc / f"temperature_{args.temperature}" / args.run / f"{doc}.txt"
                else:
                    pred_path = results_old / cat / model / doc / args.run / f"{doc}.txt"
                if not pred_path.is_file():
                    continue
                all_gt.append(gt_path.read_text(encoding="utf-8"))
                all_hyp.append(pred_path.read_text(encoding="utf-8"))

            if all_gt:
                full_gt = "\n".join(all_gt)
                full_hyp = "\n".join(all_hyp)
                cer, wer = compute_txt_metrics(full_gt, full_hyp, normalized=norm)
                ocr_data[(model, "Full Sample", norm)] = (cer, wer)

    # Build Table A (CER + WER, matching paper style with \downarrow headers)
    n_models = len(ocr_models)
    col_spec = "l" + " rr" * n_models
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{OCR and OCR Post-Correction Results for Revision Models}")
    lines.append(r"\label{tab:ocr-results-revision}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: model names with multicolumn
    header1 = " "
    for _, _, label in ocr_models:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{label}}}"
    header1 += r" \\"
    lines.append(header1)

    # cmidrules
    for i in range(n_models):
        start = 2 + i * 2
        end = start + 1
        lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    # Header row 2: CER/WER
    header2 = r"\textbf{Source}"
    for _ in ocr_models:
        header2 += r" & \textbf{CER} & \textbf{WER}"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    # Normalized section
    lines.append(r"\multicolumn{" + str(1 + n_models * 2) + r"}{c}{\textbf{Normalized}} \\")
    lines.append(r"\midrule")

    for doc in DOC_NAMES:
        row = DOC_LABELS[doc]
        for _, model, _ in ocr_models:
            vals = ocr_data.get((model, doc, True))
            if vals is None:
                row += " & - & -"
            else:
                cer, wer = vals
                row += f" & {fmt_pct(cer)} & {fmt_pct(wer)}"
        row += r" \\"
        lines.append(row)

    # Full sample row (normalized)
    lines.append(r"\midrule")
    row = r"\textbf{Full Sample}"
    for _, model, _ in ocr_models:
        vals = ocr_data.get((model, "Full Sample", True))
        if vals is None:
            row += " & - & -"
        else:
            cer, wer = vals
            row += f" & {fmt_pct(cer)} & {fmt_pct(wer)}"
    row += r" \\"
    lines.append(row)

    # Non-normalized section
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(1 + n_models * 2) + r"}{c}{\textbf{Non-normalized}} \\")
    lines.append(r"\midrule")

    for doc in DOC_NAMES:
        row = DOC_LABELS[doc]
        for _, model, _ in ocr_models:
            vals = ocr_data.get((model, doc, False))
            if vals is None:
                row += " & - & -"
            else:
                cer, wer = vals
                row += f" & {fmt_pct(cer)} & {fmt_pct(wer)}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\midrule")
    row = r"\textbf{Full Sample}"
    for _, model, _ in ocr_models:
        vals = ocr_data.get((model, "Full Sample", False))
        if vals is None:
            row += " & - & -"
        else:
            cer, wer = vals
            row += f" & {fmt_pct(cer)} & {fmt_pct(wer)}"
    row += r" \\"
    lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append("}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes}: ")
    lines.append(r"Gemini 3.1 Pro Preview uses temperature 0.0 with dynamic thinking (thinking\_budget=-1). ")
    lines.append(r"GPT-5.5 is a reasoning model with reasoning\_effort=``high'' (temperature fixed at 1.0). ")
    lines.append(r"Post-correction uses existing Transkribus Print M1 OCR output as input. ")
    lines.append(r"CER = Character Error Rate, WER = Word Error Rate.")
    lines.append(r"\end{table*}")
    lines.append("")

    # =========================================================================
    # TABLE B: NER Results by Type (Strict + Fuzzy)
    # =========================================================================
    logger.info("Computing NER metrics...")

    ner_models = [
        ("llm_txt2csv", "gemini-3.1", "Gemini 3.1 Pro", "(GT TXT-to-CSV)"),
        ("llm_img2csv", "gemini-3.1", "Gemini 3.1 Pro", "(PNG-to-CSV)"),
        ("llm_txt2csv", "gpt-5.5", "GPT-5.5", "(GT TXT-to-CSV)"),
        ("llm_img2csv", "gpt-5.5", "GPT-5.5", "(PNG-to-CSV)"),
    ]

    # Collect: {(cat, model, doc): (strict_pct, fuzzy_pct, per_col_strict, per_col_fuzzy) or None}
    ner_data = {}
    for cat, model, _, _ in ner_models:
        for doc in DOC_NAMES:
            gt_path = gt_csv_dir / f"{doc}.csv"
            gt_df = load_csv(str(gt_path))

            if cat == "llm_txt2csv":
                pred_path = results_rev / cat / model / doc / f"temperature_{args.temperature}" / args.run / f"{doc}.csv"
            else:
                pred_path = results_rev / cat / model / doc / f"temperature_{args.temperature}" / args.run / f"{doc}.csv"

            pred_df = load_csv(str(pred_path))
            result = compute_csv_metrics(gt_df, pred_df)
            ner_data[(cat, model, doc)] = result

    # Build Table B
    n_ner = len(ner_models)
    col_spec_ner = "l r" + " rr" * n_ner
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{NER Results for Revision Models (pass@1)}")
    lines.append(r"\label{tab:ner-results-revision}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec_ner + "}")
    lines.append(r"\toprule")

    # Header row 1
    h1 = r"\multicolumn{2}{c}{}"
    for _, _, label, _ in ner_models:
        h1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}"
    h1 += r" \\"
    lines.append(h1)

    # Header row 2 (task type)
    h2 = " & "
    for _, _, _, task in ner_models:
        h2 += f" & \\multicolumn{{2}}{{c}}{{{task}}}"
    h2 += r" \\"
    lines.append(h2)

    # cmidrules
    for i in range(n_ner):
        start = 3 + i * 2
        end = start + 1
        lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    # Header row 3
    h3 = r"\textbf{Source} & \textbf{\# Cells}"
    for _ in ner_models:
        h3 += r" & \textbf{Strict} & \textbf{Fuzzy}"
    h3 += r" \\"
    lines.append(h3)
    lines.append(r"\midrule")

    # Data rows
    total_cells_all = 0
    agg_strict = {(c, m): (0, 0) for c, m, _, _ in ner_models}  # (matches, total)
    agg_fuzzy = {(c, m): (0, 0) for c, m, _, _ in ner_models}

    for doc in DOC_NAMES:
        gt_path = gt_csv_dir / f"{doc}.csv"
        gt_df = load_csv(str(gt_path))
        n_cells = gt_df.shape[0] * len(EXPECTED_COLUMNS) if gt_df is not None else 0
        total_cells_all += n_cells

        row = f"{DOC_LABELS[doc]} & {n_cells}"
        for cat, model, _, _ in ner_models:
            result = ner_data.get((cat, model, doc))
            if result is None:
                row += " & - & -"
            else:
                strict, fuzzy, _, _ = result
                row += f" & {fmt_pct(strict)} & {fmt_pct(fuzzy)}"
        row += r" \\"
        lines.append(row)

    # Full sample row
    lines.append(r"\midrule")
    row = f"\\textbf{{Full Sample}} & {total_cells_all}"
    for cat, model, _, _ in ner_models:
        # Aggregate over all docs where we have results
        total_strict = 0
        total_fuzzy = 0
        total_cells = 0
        for doc in DOC_NAMES:
            result = ner_data.get((cat, model, doc))
            if result is None:
                continue
            gt_df = load_csv(str(gt_csv_dir / f"{doc}.csv"))
            if gt_df is None:
                continue
            n = gt_df.shape[0] * len(EXPECTED_COLUMNS)
            strict, fuzzy, _, _ = result
            total_strict += strict * n / 100
            total_fuzzy += fuzzy * n / 100
            total_cells += n

        if total_cells > 0:
            row += f" & {fmt_pct(total_strict / total_cells * 100)} & {fmt_pct(total_fuzzy / total_cells * 100)}"
        else:
            row += " & - & -"
    row += r" \\"
    lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append("}")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\begin{minipage}{\textwidth}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes}: Strict and fuzzy match rates (\%) for all directories. "
                 r"Before evaluation, all cells were lowercased and leading/trailing whitespaces removed. "
                 r"For fuzzy matches, Jaro-Winkler similarity with threshold 0.9 was applied. "
                 r"Gemini 3.1 Pro Preview: temperature 0.0, dynamic thinking. "
                 r"GPT-5.5: reasoning\_effort=``high'', temperature=default (1.0). "
                 r"``-'' indicates dimension mismatch (predicted row count $\neq$ ground truth).")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table*}")
    lines.append("")

    # =========================================================================
    # TABLE C: NER Results by Variable
    # =========================================================================
    logger.info("Computing NER by-variable metrics...")

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{NER Results by Variables for Revision Models (pass@1)}")
    lines.append(r"\label{tab:ner-by-variables-revision}")
    lines.append(r"\vspace{0.75em}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + " rr" * n_ner + "}")
    lines.append(r"\toprule")

    # Headers (same model structure)
    h1 = " "
    for _, _, label, _ in ner_models:
        h1 += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{label}}}}}"
    h1 += r" \\"
    lines.append(h1)

    h2 = " "
    for _, _, _, task in ner_models:
        h2 += f" & \\multicolumn{{2}}{{c}}{{{task}}}"
    h2 += r" \\"
    lines.append(h2)

    for i in range(n_ner):
        start = 2 + i * 2
        end = start + 1
        lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    h3 = r"\textbf{Variable}"
    for _ in ner_models:
        h3 += r" & \textbf{Strict} & \textbf{Fuzzy}"
    h3 += r" \\"
    lines.append(h3)
    lines.append(r"\midrule")

    # Aggregate per-column metrics across all matched docs
    for col in EXPECTED_COLUMNS:
        row = col.replace("first and middle names", "First names").replace("surname", "Surnames").replace("occupation", "Occupation").replace("address", "Address")
        for cat, model, _, _ in ner_models:
            col_strict_sum = 0.0
            col_fuzzy_sum = 0.0
            col_total = 0
            for doc in DOC_NAMES:
                result = ner_data.get((cat, model, doc))
                if result is None:
                    continue
                gt_df = load_csv(str(gt_csv_dir / f"{doc}.csv"))
                if gt_df is None:
                    continue
                _, _, per_col_strict, per_col_fuzzy = result
                n_rows = gt_df.shape[0]
                col_strict_sum += per_col_strict[col] * n_rows / 100
                col_fuzzy_sum += per_col_fuzzy[col] * n_rows / 100
                col_total += n_rows

            if col_total > 0:
                row += f" & {fmt_pct(col_strict_sum / col_total * 100)} & {fmt_pct(col_fuzzy_sum / col_total * 100)}"
            else:
                row += " & - & -"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("}")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\begin{minipage}{\linewidth}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes}: Aggregated strict and fuzzy match rates (\%) by variable. "
                 r"Only directories where the predicted CSV matched the ground truth row count are included. "
                 r"Gemini 3.1 Pro Preview: temperature 0.0, dynamic thinking. "
                 r"GPT-5.5: reasoning\_effort=``high'', temperature=default (1.0).")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    # Write output
    out_path = out_dir / "revision_tables.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Written to {out_path}")


if __name__ == "__main__":
    main()
