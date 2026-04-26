#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking CSV extraction accuracy for the *revisions* experiment.

Adapted from src/benchmarking/csv_accuracy.py.  Key differences:
  - Reads predictions from  results_revisions/  (not results/).
  - Only two pipeline categories: llm_img2csv, llm_txt2csv (no pdf2csv).
  - Produces LaTeX .tex table files (in addition to the HTML report).

Generates SIX result tables:

  -- EXACT MATCHING --
  Table 1 (Exact):      accuracy by document type  (type-1 .. type-10 + All)
  Table 2 (Exact):      accuracy by variable       (4 columns + All)

  -- NORMALIZED MATCHING (ASCII-only, punctuation removed, lower-cased) --
  Table 3 (Normalized): accuracy by document type
  Table 4 (Normalized): accuracy by variable

  -- FUZZY MATCHING (Jaro-Winkler >= 0.90) --
  Table 5 (Fuzzy):      accuracy by document type
  Table 6 (Fuzzy):      accuracy by variable

Only these 4 columns are compared (the "id" column is excluded):
  ["first and middle names", "surname", "occupation", "address"]

Matching rules:
  - EXACT:      cell1 == cell2 after basic lowercasing / whitespace trimming.
  - NORMALIZED: ASCII-only, punctuation removed, lower-cased, then exact compare.
  - FUZZY:      Jaro-Winkler similarity >= 0.90 counts as a match.

If the predicted CSV has a different row count than ground truth, the
document is flagged as a "dimension mismatch" and excluded from aggregates.

Usage example:
  python csv_accuracy.py --temperature 0.0 --run run_01
"""

import os
import sys
import re
import argparse
import logging
from joblib import Parallel, delayed
import pandas as pd

# Jaro-Winkler similarity from the rapidfuzz library (C++ backend).
from rapidfuzz.distance import JaroWinkler

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
# Configuration constants
##############################################################################

# The 10 document types that make up the benchmark corpus.
DOC_NAMES = [f"type-{i}" for i in range(1, 11)]  # type-1 .. type-10

# The four columns we compare (the "id" column is intentionally excluded).
EXPECTED_COLUMNS = ["first and middle names", "surname", "occupation", "address"]

# Only two pipeline categories in the revisions experiment (no pdf2csv).
CATEGORIES = ["llm_img2csv", "llm_txt2csv"]

# Jaro-Winkler threshold for fuzzy matching.
FUZZY_THRESHOLD = 0.90


##############################################################################
# CLI argument parsing
##############################################################################

def parse_arguments():
    """Parse command-line arguments for the benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Benchmark CSV extraction vs. ground truth (revisions, parallelized)."
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0.0",
        help="Temperature folder name inside the results tree (default: 0.0).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="run_01",
        help="Run folder name inside the results tree (default: run_01).",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="csv_accuracy_revisions.html",
        help="Name of the output HTML report file (default: csv_accuracy_revisions.html).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel joblib workers (-1 = all cores, default: -1).",
    )
    return parser.parse_args()


##############################################################################
# Cell normalization helpers
##############################################################################

def normalize_cell_value(x: str) -> str:
    """
    Basic normalization used for the EXACT matching step.

    Rules:
      - None  -> ""
      - Strip whitespace, convert to lower case.
      - Empty string or literal "null" -> "".
    """
    if x is None:
        return ""
    x_str = str(x).strip().lower()
    if x_str in ("", "null"):
        return ""
    return x_str


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalize_cell_value to every cell in the DataFrame (in-place)."""
    for col in df.columns:
        df[col] = df[col].apply(normalize_cell_value)
    return df


def ascii_only_punct_removed_lower(x: str) -> str:
    """
    Aggressive normalization used for the NORMALIZED matching step.

    Pipeline:
      1. Lower-case the string.
      2. Drop all non-ASCII characters.
      3. Remove punctuation (anything that is not alphanumeric, underscore,
         or whitespace).
      4. Collapse multiple spaces into one and strip.
    """
    if not x:
        return ""
    # Step 1 -- lower-case
    x = x.lower()
    # Step 2 -- drop non-ASCII bytes
    x = x.encode("ascii", "ignore").decode("ascii", errors="ignore")
    # Step 3 -- remove punctuation
    x = re.sub(r"[^\w\s]", "", x)
    # Step 4 -- collapse whitespace
    x = " ".join(x.split())
    return x


##############################################################################
# CSV loading and column filtering
##############################################################################

def load_csv_safely(path: str) -> pd.DataFrame | None:
    """
    Load a CSV file into a pandas DataFrame.

    Returns None if the file is missing or cannot be parsed.  All values are
    read as strings to avoid unintended type coercion.
    """
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
        return df
    except Exception as e:
        logger.warning("Could not load CSV at %s: %s", path, e)
        return None


def filter_expected_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Retain only the EXPECTED_COLUMNS in the given order.

    Returns None (treated as a dimension mismatch) if the DataFrame is None
    or any expected column is missing.  Also applies basic cell normalization.
    """
    if df is None:
        return None
    # Verify every expected column is present.
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            return None
    # Keep only the expected columns in order and drop extras.
    df = df.reindex(columns=EXPECTED_COLUMNS)
    # Normalize cells for the EXACT comparison baseline.
    df = normalize_dataframe(df)
    return df


##############################################################################
# Matching / comparison functions
##############################################################################

def compare_dataframes_exact(gt_df, pred_df):
    """
    EXACT matching: cell-by-cell equality after basic normalization.

    If row counts differ the document is flagged as a dimension mismatch.

    Returns:
        (matches, total_cells, is_mismatch, predicted_row_count)
    """
    if gt_df is None or pred_df is None:
        return (0, 0, True, 0)

    gt_rows = gt_df.shape[0]
    pred_rows = pred_df.shape[0]

    # Row-count mismatch => skip this document.
    if gt_rows != pred_rows:
        return (0, 0, True, pred_rows)

    # Element-wise equality across all 4 columns.
    matches = int((gt_df.values == pred_df.values).sum())
    total = gt_rows * len(EXPECTED_COLUMNS)
    return (matches, total, False, pred_rows)


def compare_dataframes_normalized(gt_df, pred_df):
    """
    NORMALIZED matching: ascii_only_punct_removed_lower is applied to both
    sides before comparing for equality.

    Returns:
        (matches, total_cells, is_mismatch, predicted_row_count)
    """
    if gt_df is None or pred_df is None:
        return (0, 0, True, 0)

    gt_rows = gt_df.shape[0]
    pred_rows = pred_df.shape[0]

    if gt_rows != pred_rows:
        return (0, 0, True, pred_rows)

    total_cells = gt_rows * len(EXPECTED_COLUMNS)
    match_count = 0

    # Iterate cell-by-cell and apply the normalized pipeline to both values.
    for row_idx in range(gt_rows):
        for col_idx in range(len(EXPECTED_COLUMNS)):
            gt_val = gt_df.iat[row_idx, col_idx]
            pr_val = pred_df.iat[row_idx, col_idx]
            if ascii_only_punct_removed_lower(gt_val) == ascii_only_punct_removed_lower(pr_val):
                match_count += 1

    return (match_count, total_cells, False, pred_rows)


def compare_dataframes_fuzzy(gt_df, pred_df, threshold=FUZZY_THRESHOLD):
    """
    FUZZY matching: Jaro-Winkler similarity >= threshold counts as a match.

    A minimal strip/lower is applied before computing similarity so that
    trivial casing or whitespace differences do not penalize the score.

    Returns:
        (matches, total_cells, is_mismatch, predicted_row_count)
    """
    if gt_df is None or pred_df is None:
        return (0, 0, True, 0)

    gt_rows = gt_df.shape[0]
    pred_rows = pred_df.shape[0]

    if gt_rows != pred_rows:
        return (0, 0, True, pred_rows)

    total_cells = gt_rows * len(EXPECTED_COLUMNS)
    match_count = 0

    for row_idx in range(gt_rows):
        for col_idx in range(len(EXPECTED_COLUMNS)):
            val_gt = gt_df.iat[row_idx, col_idx].strip().lower()
            val_pr = pred_df.iat[row_idx, col_idx].strip().lower()
            sim = JaroWinkler.similarity(val_gt, val_pr)
            if sim >= threshold:
                match_count += 1

    return (match_count, total_cells, False, pred_rows)


##############################################################################
# LaTeX table generation helpers
##############################################################################

def _escape_latex(text: str) -> str:
    """Escape characters that have special meaning in LaTeX."""
    # Replace underscores with \_  and  % with \%  etc.
    text = text.replace("_", r"\_")
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("#", r"\#")
    return text


def build_table1_latex(table1_dict, gt_row_counts, table_number, method_label):
    """
    Build a LaTeX table string for a "by type" table.

    Columns: Model | type-1 | type-2 | ... | type-10 | All
    Each cell shows accuracy as a percentage (e.g. 92.60%).
    Dimension mismatches are shown as "mismatch".
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        rf"\caption{{Table {table_number} ({method_label}): By Type."
        r" Gemini-3.1 uses temperature 0.0."
        r" GPT-5.5 is a reasoning model (temperature fixed at 1.0).}"
    )
    # Build the tabular environment inside a resizebox for wide tables.
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column spec: l for model name, then 10 c columns for types + 1 c for All.
    col_spec = "l|" + "c" * 10 + "|c"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")

    # Header row with document type names.
    header_parts = ["Model"]
    for doc in DOC_NAMES:
        header_parts.append(_escape_latex(doc))
    header_parts.append("All")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\hline")

    # One row per model, sorted alphabetically.
    sorted_mks = sorted(table1_dict.keys(), key=lambda x: x.lower())
    for mk in sorted_mks:
        row_parts = [_escape_latex(mk)]
        for doc in DOC_NAMES + ["All"]:
            matches, total, mismatch, pred_nrows = table1_dict[mk][doc]
            if mismatch and doc != "All":
                # Dimension mismatch for a specific document.
                cell_val = "mismatch"
            elif mismatch and doc == "All":
                cell_val = "0.00\\%"
            else:
                if total > 0:
                    rate = (matches / total) * 100.0
                    cell_val = f"{rate:.2f}\\%"
                else:
                    cell_val = "0.00\\%"
            row_parts.append(cell_val)
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\hline")

    # Footer row: total number of cells per document (GT rows * 4).
    footer_parts = ["Total cells"]
    for doc in DOC_NAMES:
        r = gt_row_counts.get(doc, 0)
        footer_parts.append(str(r * len(EXPECTED_COLUMNS)))
    sum_r = sum(gt_row_counts.get(d, 0) for d in DOC_NAMES)
    footer_parts.append(str(sum_r * len(EXPECTED_COLUMNS)))
    lines.append(" & ".join(footer_parts) + r" \\")

    # Footer row: GT row counts.
    gt_parts = ["GT Rows"]
    for doc in DOC_NAMES:
        gt_parts.append(str(gt_row_counts.get(doc, 0)))
    gt_parts.append(str(sum_r))
    lines.append(" & ".join(gt_parts) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("}")  # close resizebox
    lines.append(rf"\label{{tab:revision_table{table_number}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_table2_latex(table2_final_dict, table_number, method_label):
    """
    Build a LaTeX table string for a "by variable" table.

    Columns: Model | first and middle names | surname | occupation | address | All
    Each cell shows "XX.XX% (hits/total)".
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        rf"\caption{{Table {table_number} ({method_label}): By Variable."
        r" Gemini-3.1 uses temperature 0.0."
        r" GPT-5.5 is a reasoning model (temperature fixed at 1.0).}"
    )
    lines.append(r"\resizebox{\textwidth}{!}{%")

    # Column spec: l for model + 4 variable columns + All.
    col_spec = "l|" + "c" * len(EXPECTED_COLUMNS) + "|c"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")

    # Header row.
    header_parts = ["Model"]
    for col in EXPECTED_COLUMNS:
        header_parts.append(_escape_latex(col))
    header_parts.append("All")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\hline")

    # One row per model.
    sorted_mks = sorted(table2_final_dict.keys(), key=lambda x: x.lower())
    for mk in sorted_mks:
        row_parts = [_escape_latex(mk)]
        for col in EXPECTED_COLUMNS + ["All"]:
            m, t = table2_final_dict[mk][col]
            if t > 0:
                rate = (m / t) * 100.0
                cell_val = f"{rate:.2f}\\% ({m}/{t})"
            else:
                cell_val = "0.00\\% (0/0)"
            row_parts.append(cell_val)
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("}")  # close resizebox
    lines.append(rf"\label{{tab:revision_table{table_number}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


##############################################################################
# HTML table generation helpers  (carried over from the original script)
##############################################################################

def build_table1_html(table1_dict, gt_row_counts, table_title):
    """
    Build an HTML table for a "by type" result set.

    Each cell shows "XX.XX% | #matched_cells" or "dim mismatch (r=N)".
    """
    out = []
    out.append(f"<h2>{table_title}</h2>")
    out.append("<table><tr>")
    out.append("<th>Model</th>")
    for doc in DOC_NAMES:
        out.append(f"<th>{doc}</th>")
    out.append("<th>All</th>")
    out.append("</tr>")

    sorted_mks = sorted(table1_dict.keys(), key=lambda x: x.lower())
    for mk in sorted_mks:
        out.append(f"<tr><td class='model-name'>{mk}</td>")
        for doc in DOC_NAMES + ["All"]:
            matches, total, mismatch, pred_nrows = table1_dict[mk][doc]
            if mismatch and doc != "All":
                cell_val = f"<span class='mismatch'>dim mismatch (r={pred_nrows})</span>"
            elif mismatch and doc == "All":
                cell_val = "<span class='mismatch'>0.00% | 0</span>"
            else:
                if total > 0:
                    rate = (matches / total) * 100.0
                    cell_val = f"{rate:.2f}% | {matches}"
                else:
                    cell_val = "0.00% | 0"
            out.append(f"<td>{cell_val}</td>")
        out.append("</tr>")

    # Footer: total number of cells = GT rows * 4.
    out.append("<tr><td class='model-name'>Total number of cells</td>")
    for doc in DOC_NAMES:
        r = gt_row_counts.get(doc, 0)
        out.append(f"<td>{r * len(EXPECTED_COLUMNS)}</td>")
    sum_r = sum(gt_row_counts.get(d, 0) for d in DOC_NAMES)
    out.append(f"<td>{sum_r * len(EXPECTED_COLUMNS)}</td>")
    out.append("</tr>")

    # Footer: GT row counts.
    out.append("<tr><td class='model-name'>GT Rows</td>")
    for doc in DOC_NAMES:
        out.append(f"<td>{gt_row_counts.get(doc, 0)}</td>")
    out.append(f"<td>{sum_r}</td>")
    out.append("</tr>")

    out.append("</table>")
    return "\n".join(out)


def build_table2_html(table2_final_dict, table_title):
    """
    Build an HTML table for a "by variable" result set.

    Each cell shows "XX.XX% (hits/total)".
    """
    out = []
    out.append(f"<h2>{table_title}</h2>")
    out.append("<table><tr>")
    out.append("<th>Model</th>")
    for col in EXPECTED_COLUMNS:
        out.append(f"<th>{col}</th>")
    out.append("<th>All</th>")
    out.append("</tr>")

    sorted_mks = sorted(table2_final_dict.keys(), key=lambda x: x.lower())
    for mk in sorted_mks:
        out.append(f"<tr><td class='model-name'>{mk}</td>")
        for col in EXPECTED_COLUMNS + ["All"]:
            m, t = table2_final_dict[mk][col]
            if t > 0:
                rate = (m / t) * 100.0
                cell_val = f"{rate:.2f}% ({m}/{t})"
            else:
                cell_val = "0.00% (0/0)"
            out.append(f"<td>{cell_val}</td>")
        out.append("</tr>")

    out.append("</table>")
    return "\n".join(out)


##############################################################################
# Main entry point
##############################################################################

def main():
    args = parse_arguments()

    # Resolve project paths relative to this script's location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    logger.info("Project root: %s", project_root)

    # Ground truth lives in data/ground_truth/csv/.
    ground_truth_dir = os.path.join(project_root, "data", "ground_truth", "csv")

    # Predictions live in results_revisions/ (NOT results/).
    results_dir = os.path.join(project_root, "results_revisions")

    # Directory where LaTeX .tex files will be written.
    tables_dir = os.path.join(script_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load ground-truth DataFrames and record row counts.
    # ------------------------------------------------------------------
    gt_dataframes = {}   # doc -> filtered DataFrame (or None)
    gt_row_counts = {}   # doc -> int

    for doc in DOC_NAMES:
        path = os.path.join(ground_truth_dir, f"{doc}.csv")
        df_original = load_csv_safely(path)
        df_filtered = filter_expected_columns(df_original)
        if df_filtered is not None:
            gt_dataframes[doc] = df_filtered
            gt_row_counts[doc] = df_filtered.shape[0]
        else:
            gt_dataframes[doc] = None
            gt_row_counts[doc] = 0

    # ------------------------------------------------------------------
    # 2) Discover models by scanning results_revisions/{category}/.
    # ------------------------------------------------------------------
    discovered_models = []
    for cat in CATEGORIES:
        cat_path = os.path.join(results_dir, cat)
        if not os.path.isdir(cat_path):
            logger.info("Category directory not found, skipping: %s", cat_path)
            continue
        for model in sorted(os.listdir(cat_path)):
            model_path = os.path.join(cat_path, model)
            if os.path.isdir(model_path):
                discovered_models.append((cat, model))

    if not discovered_models:
        logger.warning("No models found under %s.", results_dir)
        return

    logger.info(
        "Discovered %d model(s): %s",
        len(discovered_models),
        [(c, m) for c, m in discovered_models],
    )

    # ------------------------------------------------------------------
    # 3) Prepare empty data structures for all 6 tables.
    # ------------------------------------------------------------------
    # "table1" = by-type tables;  "table2" = by-variable tables.
    # Each has exact / normalized / fuzzy variants.
    table1_exact = {}
    table2_exact = {}
    table1_norm = {}
    table2_norm = {}
    table1_fuzzy = {}
    table2_fuzzy = {}

    for cat, model in discovered_models:
        mk = f"{cat}/{model}"  # composite key, e.g. "llm_img2csv/gemini-3.1"
        table1_exact[mk] = {}
        table2_exact[mk] = {}
        table1_norm[mk] = {}
        table2_norm[mk] = {}
        table1_fuzzy[mk] = {}
        table2_fuzzy[mk] = {}

    # ------------------------------------------------------------------
    # 4) Define the per-document worker that will run in parallel.
    # ------------------------------------------------------------------
    def process_doc(cat, model, doc):
        """
        Compare one (model, document) pair against ground truth using all
        three matching strategies.  Returns a tuple of results that will be
        unpacked and stored in the table dictionaries.
        """
        mk = f"{cat}/{model}"

        # Ground truth for this document.
        gt_df = gt_dataframes[doc]

        # Build the path to the predicted CSV:
        #   results_revisions/{cat}/{model}/{doc}/temperature_{T}/run_{R}/{doc}.csv
        pred_path = os.path.join(
            results_dir, cat, model, doc,
            f"temperature_{args.temperature}",
            args.run,
            f"{doc}.csv",
        )
        pred_df_original = load_csv_safely(pred_path)
        pred_df_filtered = filter_expected_columns(pred_df_original)

        # --- EXACT comparison ---
        exact_matches, exact_total, exact_mismatch, exact_pred_nrows = (
            compare_dataframes_exact(gt_df, pred_df_filtered)
        )
        # Per-variable breakdown for exact matching.
        if exact_mismatch or gt_df is None or pred_df_filtered is None:
            t2_exact = None
        else:
            t2_exact = {}
            for col in EXPECTED_COLUMNS:
                gt_col = gt_df[col].values
                pr_col = pred_df_filtered[col].values
                col_match = int((gt_col == pr_col).sum())
                col_total = len(gt_col)
                t2_exact[col] = (col_match, col_total)

        # --- NORMALIZED comparison ---
        norm_matches, norm_total, norm_mismatch, norm_pred_nrows = (
            compare_dataframes_normalized(gt_df, pred_df_filtered)
        )
        # Per-variable breakdown for normalized matching.
        if norm_mismatch or gt_df is None or pred_df_filtered is None:
            t2_norm = None
        else:
            t2_norm = {}
            rows_count = gt_df.shape[0]
            for col in EXPECTED_COLUMNS:
                col_match = 0
                col_total = rows_count
                gt_col = gt_df[col].values
                pr_col = pred_df_filtered[col].values
                for i in range(rows_count):
                    if ascii_only_punct_removed_lower(gt_col[i]) == ascii_only_punct_removed_lower(pr_col[i]):
                        col_match += 1
                t2_norm[col] = (col_match, col_total)

        # --- FUZZY comparison ---
        fuzzy_matches, fuzzy_total, fuzzy_mismatch, fuzzy_pred_nrows = (
            compare_dataframes_fuzzy(gt_df, pred_df_filtered)
        )
        # Per-variable breakdown for fuzzy matching.
        if fuzzy_mismatch or gt_df is None or pred_df_filtered is None:
            t2_fuzzy = None
        else:
            t2_fuzzy = {}
            rows_count = gt_df.shape[0]
            for col in EXPECTED_COLUMNS:
                col_match = 0
                col_total = rows_count
                gt_col = gt_df[col].values
                pr_col = pred_df_filtered[col].values
                for i in range(rows_count):
                    sim = JaroWinkler.similarity(
                        gt_col[i].strip().lower(),
                        pr_col[i].strip().lower(),
                    )
                    if sim >= FUZZY_THRESHOLD:
                        col_match += 1
                t2_fuzzy[col] = (col_match, col_total)

        # Pack all results into a single tuple for the parallel collector.
        return (
            mk, doc,
            (exact_matches, exact_total, exact_mismatch, exact_pred_nrows), t2_exact,
            (norm_matches, norm_total, norm_mismatch, norm_pred_nrows), t2_norm,
            (fuzzy_matches, fuzzy_total, fuzzy_mismatch, fuzzy_pred_nrows), t2_fuzzy,
        )

    # ------------------------------------------------------------------
    # 5) Run all (model, document) pairs in parallel via joblib.
    # ------------------------------------------------------------------
    tasks = []
    for cat, model in discovered_models:
        for doc in DOC_NAMES:
            tasks.append((cat, model, doc))

    logger.info("Launching %d parallel tasks (n_jobs=%s)...", len(tasks), args.n_jobs)
    parallel_results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_doc)(cat, model, doc) for (cat, model, doc) in tasks
    )

    # ------------------------------------------------------------------
    # 6) Collect parallel results back into the table dictionaries.
    # ------------------------------------------------------------------
    for item in parallel_results:
        (mk, doc,
         t1e_data, t2e_dict,
         t1n_data, t2n_dict,
         t1f_data, t2f_dict) = item

        table1_exact[mk][doc] = t1e_data
        table2_exact[mk][doc] = t2e_dict

        table1_norm[mk][doc] = t1n_data
        table2_norm[mk][doc] = t2n_dict

        table1_fuzzy[mk][doc] = t1f_data
        table2_fuzzy[mk][doc] = t2f_dict

    # ------------------------------------------------------------------
    # 7) Aggregate results across all documents for the "All" column.
    # ------------------------------------------------------------------
    def aggregate_table1(table1_dict):
        """
        For each model, sum up matches and totals across all non-mismatched
        documents to produce the "All" column.
        """
        for mk in table1_dict:
            sum_m = 0   # total matched cells
            sum_t = 0   # total compared cells
            any_valid = False
            for doc in DOC_NAMES:
                m, t, mm, _ = table1_dict[mk][doc]
                if not mm:
                    sum_m += m
                    sum_t += t
                    any_valid = True
            if not any_valid:
                table1_dict[mk]["All"] = (0, 0, True, 0)
            else:
                table1_dict[mk]["All"] = (sum_m, sum_t, False, 0)

    def aggregate_table2(table2_dict):
        """
        For each model, sum per-variable matches across all non-mismatched
        documents.  Returns a new dict: final[mk][col] = [sum_m, sum_t].
        """
        final = {}
        for mk in table2_dict:
            final[mk] = {}
            for c in EXPECTED_COLUMNS + ["All"]:
                final[mk][c] = [0, 0]
            for doc in DOC_NAMES:
                doc_dict = table2_dict[mk][doc]
                if doc_dict is None:
                    continue  # skip mismatched documents
                for c in EXPECTED_COLUMNS:
                    cm, ct = doc_dict[c]
                    final[mk][c][0] += cm
                    final[mk][c][1] += ct
                    final[mk]["All"][0] += cm
                    final[mk]["All"][1] += ct
        return final

    # Aggregate the three "by type" tables.
    for tb in (table1_exact, table1_norm, table1_fuzzy):
        aggregate_table1(tb)

    # Aggregate the three "by variable" tables.
    table2_exact_final = aggregate_table2(table2_exact)
    table2_norm_final = aggregate_table2(table2_norm)
    table2_fuzzy_final = aggregate_table2(table2_fuzzy)

    # ------------------------------------------------------------------
    # 8) Generate LaTeX .tex files for all 6 tables.
    # ------------------------------------------------------------------
    logger.info("Generating LaTeX tables in %s ...", tables_dir)

    latex_tables = [
        # (table_number, method_label, builder_func, data_dict)
        (1, "Exact", "type", table1_exact),
        (2, "Exact", "var", table2_exact_final),
        (3, "Normalized", "type", table1_norm),
        (4, "Normalized", "var", table2_norm_final),
        (5, "Fuzzy", "type", table1_fuzzy),
        (6, "Fuzzy", "var", table2_fuzzy_final),
    ]

    for tbl_num, method, kind, data in latex_tables:
        if kind == "type":
            latex_str = build_table1_latex(data, gt_row_counts, tbl_num, method)
        else:
            latex_str = build_table2_latex(data, tbl_num, method)

        tex_filename = f"csv_table{tbl_num}_{method.lower()}_{kind}.tex"
        tex_path = os.path.join(tables_dir, tex_filename)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        logger.info("  Wrote %s", tex_path)

    # ------------------------------------------------------------------
    # 9) Generate the HTML report (same format as the original script).
    # ------------------------------------------------------------------
    logger.info("Generating HTML report...")

    html_head = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Benchmark CSV Extraction (Revisions)</title>
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
  margin-bottom: 0.5em;
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
}
tr:nth-child(even) {
  background: #f2f2f2;
}
.model-name {
  font-weight: bold;
  background: #f9f9de !important;
}
.mismatch {
  color: red;
  font-weight: bold;
}
.note {
  font-size: 0.9em;
  line-height: 1.4em;
  color: #555;
  border-top: 2px solid #ccc;
  padding-top: 0.5em;
  margin-bottom: 1.5em;
}
.section-note {
  font-size: 0.9em;
  line-height: 1.4em;
  color: #555;
  margin-bottom: 2em;
}
</style>
</head>
<body>
<h1>CSV Extraction Benchmark &mdash; Revisions (Exact, Normalized, Fuzzy)</h1>
"""

    html_body = []

    # Introductory notes explaining the methodology.
    html_body.append(f"""
<p><strong>General Remarks (Revisions Benchmark):</strong><br>
We compare predictions from <code>results_revisions/</code> to ground truth
using exactly 4 columns: {EXPECTED_COLUMNS}.<br>
Row mismatches &rArr; <em>dim mismatch</em>.<br>
Empty, "null", or whitespace cells are normalized to "" for all methods.<br><br>

<b>Matching rules:</b><br>
1) <em>Exact</em>: direct string equality (lowercased, trimmed).<br>
2) <em>Normalized</em>: ASCII-only, punctuation-removed, then exact compare.<br>
3) <em>Fuzzy</em>: Jaro-Winkler similarity &ge; {FUZZY_THRESHOLD}.<br><br>

<b>Total number of cells</b>: (GT row count) &times; 4.<br>
<b>GT Rows</b>: ground-truth row count for each doc.<br>
</p>
""")

    # --- EXACT Tables (1 & 2) ---
    html_body.append(build_table1_html(table1_exact, gt_row_counts, "Table 1 (Exact): By Type"))
    html_body.append("""
<div class="note">
  <strong>Notes for Table 1 (Exact):</strong>
  <ul>
    <li>Cells are "XX.XX% | #hits" if dimension matches, otherwise "dim mismatch (r=NN)".</li>
    <li>"Total number of cells" = GT row count * 4.</li>
    <li>"GT Rows" is the ground-truth row count for each doc.</li>
  </ul>
</div>
""")

    html_body.append(build_table2_html(table2_exact_final, "Table 2 (Exact): By Variable"))
    html_body.append("""
<div class="note">
  <strong>Notes for Table 2 (Exact):</strong>
  <ul>
    <li>Mismatched docs are skipped. Each cell shows "XX.XX% (#hits / #total)".</li>
    <li>If a model has zero matched docs, it's "0.00% (0/0)".</li>
  </ul>
</div>
""")

    # --- NORMALIZED Tables (3 & 4) ---
    html_body.append(build_table1_html(table1_norm, gt_row_counts, "Table 3 (Normalized): By Type"))
    html_body.append("""
<div class="note">
  <strong>Notes for Table 3 (Normalized):</strong>
  <ul>
    <li>Removes punctuation and non-ASCII, then lower-cases before direct comparison.</li>
    <li>Dimension mismatch is the same concept: row counts must match.</li>
  </ul>
</div>
""")

    html_body.append(build_table2_html(table2_norm_final, "Table 4 (Normalized): By Variable"))
    html_body.append("""
<div class="note">
  <strong>Notes for Table 4 (Normalized):</strong>
  <ul>
    <li>Mismatched docs are skipped. Each cell shows "XX.XX% (#hits / #total)".</li>
    <li>This helps catch cells that only differ by accents or punctuation.</li>
  </ul>
</div>
""")

    # --- FUZZY Tables (5 & 6) ---
    html_body.append(build_table1_html(table1_fuzzy, gt_row_counts, "Table 5 (Fuzzy): By Type"))
    html_body.append(f"""
<div class="note">
  <strong>Notes for Table 5 (Fuzzy):</strong>
  <ul>
    <li>A cell is considered a match if Jaro-Winkler similarity &ge; {FUZZY_THRESHOLD}.</li>
    <li>Dimension mismatch is the same concept: row counts must match.</li>
    <li>We do minimal lowercasing/whitespace-trimming before measuring similarity.</li>
  </ul>
</div>
""")

    html_body.append(build_table2_html(table2_fuzzy_final, "Table 6 (Fuzzy): By Variable"))
    html_body.append(f"""
<div class="note">
  <strong>Notes for Table 6 (Fuzzy):</strong>
  <ul>
    <li>Mismatched docs are skipped. Each cell shows "XX.XX% (#hits / #total)".</li>
    <li>We consider two cells a match if Jaro-Winkler similarity &ge; {FUZZY_THRESHOLD}.</li>
  </ul>
</div>
""")

    # Overall implementation-detail footer.
    used_cores = args.n_jobs if args.n_jobs != -1 else os.cpu_count()
    html_body.append(f"""
<div class="section-note">
  <strong>Implementation Details:</strong><br>
  - Parallelized with joblib, n_jobs={used_cores}.<br>
  - Fuzzy matching via <code>rapidfuzz.distance.JaroWinkler.similarity()</code>
    (&ge; {FUZZY_THRESHOLD} &rArr; match).<br>
  - Normalization: ASCII-only, punctuation removed, lower-cased.<br>
  - Predictions read from <code>results_revisions/</code>.<br>
  - Categories: {CATEGORIES}.<br>
  - LaTeX tables saved to <code>{tables_dir}</code>.<br>
</div>
</body>
</html>
""")

    # Write the final HTML file next to this script.
    final_html = html_head + "\n".join(html_body)
    out_path = os.path.join(script_dir, args.output_html)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    logger.info(
        "Done!  HTML report saved to '%s'.  LaTeX tables in '%s'.",
        out_path,
        tables_dir,
    )


if __name__ == "__main__":
    main()
