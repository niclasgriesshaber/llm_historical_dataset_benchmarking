#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking LLM vs. PDF/IMG/TXT CSV extraction, parallelized with joblib,
generating SIX tables now:

 -- EXACT MATCHING --
 1) Table 1 (Exact) by Type
 2) Table 2 (Exact) by Variable

 -- FUZZY MATCHING (Jaro-Winkler, threshold=0.90) --
 3) Table 3 (Fuzzy) by Type
 4) Table 4 (Fuzzy) by Variable

 -- LEVENSHTEIN-BASED MATCHING --
 5) Table 5 (Levenshtein): By Type
 6) Table 6 (Levenshtein): By Variable

We only allow these 4 columns in the final comparison:
  ["first and middle names", "surname", "occupation", "address"]

(Previously included "id", but we now exclude it from all calculations.)

However, for Table 1's row "Total number of cells", we use the GROUND-TRUTH CSV's
original column count (before we remove extras) multiplied by the GT row count.
The row "GT Rows" is just the ground-truth row count for each doc.

Matching rules:
 - EXACT: cell1 == cell2 after basic normalization (empty, "null" => "")
 - FUZZY: Jaro-Winkler similarity >= 0.90 => matched
 - LEVENSHTEIN: distance <= (number of words in the GT cell)

If row counts differ, we label dimension mismatch (and skip that doc in the aggregated tables).
"""

import os
import sys
import argparse
import logging
from joblib import Parallel, delayed
import pandas as pd

# For fuzzy matching:
from rapidfuzz.distance import JaroWinkler

# For Levenshtein-based distance:
from rapidfuzz.distance import Levenshtein

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


##############################################################################
# Configuration
##############################################################################

DOC_NAMES = [f"type-{i}" for i in range(1, 11)]  # type-1..type-10

# EXCLUDE "id" FROM ALL CALCULATIONS:
EXPECTED_COLUMNS = ["first and middle names", "surname", "occupation", "address"]

# Now including "llm_txt2csv"
CATEGORIES = ["llm_img2csv", "llm_pdf2csv", "llm_txt2csv"]
FUZZY_THRESHOLD = 0.90


##############################################################################
# Parse command-line arguments
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark CSV extraction vs. ground truth (parallelized)."
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0.0",
        help="Temperature folder (default: 0.0)."
    )
    parser.add_argument(
        "--run",
        type=str,
        default="run_01",
        help="Run folder (default: run_01)."
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default="benchmark_csv.html",
        help="Output HTML file (default: benchmark_csv.html)."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1)."
    )
    return parser.parse_args()


##############################################################################
# Cell Normalization
##############################################################################

def normalize_cell_value(x: str) -> str:
    """
    Convert 'null', 'NULL', None, empty string, or all-whitespace => ""
    Everything else => strip leading/trailing whitespace.
    """
    if x is None:
        return ""
    x_str = str(x).strip().lower()
    if x_str in ("", "null"):
        return ""
    # Otherwise, return original but stripped
    return str(x).strip()

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    In-place: normalize every cell with normalize_cell_value.
    Return the same DF for convenience.
    """
    for col in df.columns:
        df[col] = df[col].apply(normalize_cell_value)
    return df


##############################################################################
# Helper Functions
##############################################################################

def load_csv_safely(path):
    """Load CSV into a DataFrame, or return None if missing/error."""
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
        return df
    except Exception as e:
        logger.warning(f"Could not load CSV at {path}: {e}")
        return None

def filter_expected_columns(df):
    """
    Keep only the EXPECTED_COLUMNS, in that exact order.
    If df is None or missing any of those columns, return None => mismatch.
    Otherwise, normalize every cell to handle empty/null matching.
    """
    if df is None:
        return None
    # Check if all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            return None
    # Reindex to EXACTLY those columns => remove extras
    df = df.reindex(columns=EXPECTED_COLUMNS)
    # Normalize all cells
    df = normalize_dataframe(df)
    return df


##############################################################################
# Exact Comparison
##############################################################################

def compare_dataframes_exact(gt_df, pred_df):
    """
    EXACT MATCH:
      - If row counts differ => dimension mismatch
      - Otherwise => compare cell by cell (exact equality)
    Returns (matches, total, mismatch_bool, pred_nrows).
    """
    if gt_df is None or pred_df is None:
        return (0, 0, True, 0)

    gt_rows = gt_df.shape[0]
    pred_rows = pred_df.shape[0]

    if gt_rows != pred_rows:
        return (0, 0, True, pred_rows)

    matches = (gt_df.values == pred_df.values).sum()
    total = gt_rows * len(EXPECTED_COLUMNS)
    return (matches, total, False, pred_rows)


##############################################################################
# Fuzzy Comparison (Jaro-Winkler)
##############################################################################

def jaro_winkler_sim(a: str, b: str) -> float:
    """
    Return Jaro-Winkler similarity in [0..1].
    Higher => more similar.
    """
    return JaroWinkler.similarity(a, b)

def compare_dataframes_fuzzy(gt_df, pred_df):
    """
    FUZZY MATCH:
      - If row counts differ => dimension mismatch
      - Otherwise => for each cell: JaroWinkler.similarity >= FUZZY_THRESHOLD => 1 match
    Returns (matches, total, mismatch_bool, pred_nrows).
    """
    if gt_df is None or pred_df is None:
        return (0, 0, True, 0)

    gt_rows = gt_df.shape[0]
    pred_rows = pred_df.shape[0]

    if gt_rows != pred_rows:
        return (0, 0, True, pred_rows)

    # cell-by-cell fuzzy
    total_cells = gt_rows * len(EXPECTED_COLUMNS)
    match_count = 0
    for row_idx in range(gt_rows):
        for col_idx in range(len(EXPECTED_COLUMNS)):
            val_gt = gt_df.iat[row_idx, col_idx]
            val_pr = pred_df.iat[row_idx, col_idx]
            sim = jaro_winkler_sim(val_gt, val_pr)
            if sim >= FUZZY_THRESHOLD:
                match_count += 1

    return (match_count, total_cells, False, pred_rows)


##############################################################################
# Levenshtein-based Comparison
##############################################################################

def levenshtein_distance(a: str, b: str) -> int:
    """
    Return the Levenshtein edit distance between two strings.
    """
    return Levenshtein.distance(a, b)

def compare_dataframes_levenshtein(gt_df, pred_df):
    """
    LEVENSHTEIN MATCH:
      - If row counts differ => dimension mismatch
      - Otherwise => for each cell:
            distance = Levenshtein(gt_val, pred_val)
            words_in_gt = number of words in gt_val
            match if distance <= words_in_gt
    Returns (matches, total, mismatch_bool, pred_nrows).
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
            val_gt = gt_df.iat[row_idx, col_idx]
            val_pr = pred_df.iat[row_idx, col_idx]
            dist = levenshtein_distance(val_gt, val_pr)
            words_in_gt = len(val_gt.split()) if val_gt.strip() else 0
            if dist <= words_in_gt:
                match_count += 1

    return (match_count, total_cells, False, pred_rows)


##############################################################################
# Main
##############################################################################

def main():
    args = parse_arguments()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    logger.info("Project root: %s", project_root)

    ground_truth_dir = os.path.join(project_root, "data", "ground_truth", "csv")
    results_dir = os.path.join(project_root, "results")

    # We'll store:  orig_gt_col_count[doc] => # columns in the original ground-truth CSV (before filtering)
    orig_gt_col_count = {}
    gt_dataframes = {}
    gt_row_counts = {}

    # 1) Load ground-truth
    for doc in DOC_NAMES:
        path = os.path.join(ground_truth_dir, f"{doc}.csv")
        df_original = load_csv_safely(path)
        if df_original is not None:
            orig_gt_col_count[doc] = df_original.shape[1]
        else:
            orig_gt_col_count[doc] = 0

        df_filtered = filter_expected_columns(df_original)
        if df_filtered is not None:
            gt_dataframes[doc] = df_filtered
            gt_row_counts[doc] = df_filtered.shape[0]
        else:
            gt_dataframes[doc] = None
            gt_row_counts[doc] = 0

    # 2) Discover models
    discovered_models = []
    for cat in CATEGORIES:
        cat_path = os.path.join(results_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for model in sorted(os.listdir(cat_path)):
            model_path = os.path.join(cat_path, model)
            if os.path.isdir(model_path):
                discovered_models.append((cat, model))

    if not discovered_models:
        logger.warning("No models found under %s.", results_dir)
        return

    # We produce six data structures:
    #  - EXACT => table1_exact, table2_exact
    #  - FUZZY => table1_fuzzy, table2_fuzzy
    #  - LEVEN => table1_lev,   table2_lev
    # Each is a dict: table1_exact[mk][doc] = (matches, total, mismatch, pred_nrows), etc.
    table1_exact = {}
    table2_exact = {}
    table1_fuzzy = {}
    table2_fuzzy = {}
    table1_lev = {}
    table2_lev = {}

    # Initialize them
    for cat, model in discovered_models:
        mk = f"{cat}/{model}"
        table1_exact[mk] = {}
        table2_exact[mk] = {}
        table1_fuzzy[mk] = {}
        table2_fuzzy[mk] = {}
        table1_lev[mk] = {}
        table2_lev[mk] = {}

    # We'll do a single pass loading predicted CSV once, then do EXACT, FUZZY, LEVEN
    def process_doc(cat, model, doc):
        mk = f"{cat}/{model}"

        # Ground truth
        gt_df = gt_dataframes[doc]

        # Decide folder name depending on the category:
        if cat == "llm_txt2csv":
            doc_dir = doc  # e.g. "type-1"
        else:
            doc_dir = f"{doc}.pdf"  # e.g. "type-1.pdf"

        # Predicted path
        pred_dir = os.path.join(results_dir, cat, model, doc_dir)
        pred_path = os.path.join(
            pred_dir,
            f"temperature_{args.temperature}",
            args.run,
            f"{doc}.csv"
        )
        pred_df_original = load_csv_safely(pred_path)
        pred_df_filtered = filter_expected_columns(pred_df_original)

        # EXACT
        exact_matches, exact_total, exact_mismatch, exact_pred_nrows = compare_dataframes_exact(gt_df, pred_df_filtered)
        # Table 2 exact
        if exact_mismatch or gt_df is None or pred_df_filtered is None:
            t2_exact = None
        else:
            # per-column exact
            t2_exact = {}
            for col in EXPECTED_COLUMNS:
                gt_col = gt_df[col].values
                pr_col = pred_df_filtered[col].values
                col_match = (gt_col == pr_col).sum()
                col_total = len(gt_col)
                t2_exact[col] = (col_match, col_total)

        # FUZZY
        fuzzy_matches, fuzzy_total, fuzzy_mismatch, fuzzy_pred_nrows = compare_dataframes_fuzzy(gt_df, pred_df_filtered)
        # Table 2 fuzzy
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
                    sim = jaro_winkler_sim(gt_col[i], pr_col[i])
                    if sim >= FUZZY_THRESHOLD:
                        col_match += 1
                t2_fuzzy[col] = (col_match, col_total)

        # LEVENSHTEIN
        lev_matches, lev_total, lev_mismatch, lev_pred_nrows = compare_dataframes_levenshtein(gt_df, pred_df_filtered)
        # Table 2 leven
        if lev_mismatch or gt_df is None or pred_df_filtered is None:
            t2_lev = None
        else:
            t2_lev = {}
            rows_count = gt_df.shape[0]
            for col in EXPECTED_COLUMNS:
                col_match = 0
                col_total = rows_count
                gt_col = gt_df[col].values
                pr_col = pred_df_filtered[col].values
                for i in range(rows_count):
                    dist = levenshtein_distance(gt_col[i], pr_col[i])
                    word_count = len(gt_col[i].split()) if gt_col[i].strip() else 0
                    if dist <= word_count:
                        col_match += 1
                t2_lev[col] = (col_match, col_total)

        return (
            mk, doc,
            (exact_matches, exact_total, exact_mismatch, exact_pred_nrows), t2_exact,
            (fuzzy_matches, fuzzy_total, fuzzy_mismatch, fuzzy_pred_nrows), t2_fuzzy,
            (lev_matches, lev_total, lev_mismatch, lev_pred_nrows), t2_lev
        )

    tasks = []
    for cat, model in discovered_models:
        for doc in DOC_NAMES:
            tasks.append((cat, model, doc))

    parallel_results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_doc)(cat, model, doc) for (cat, model, doc) in tasks
    )

    # Collect
    for item in parallel_results:
        mk, doc, t1e_data, t2e_dict, t1f_data, t2f_dict, t1l_data, t2l_dict = item
        table1_exact[mk][doc] = t1e_data
        table2_exact[mk][doc] = t2e_dict
        table1_fuzzy[mk][doc] = t1f_data
        table2_fuzzy[mk][doc] = t2f_dict
        table1_lev[mk][doc] = t1l_data
        table2_lev[mk][doc] = t2l_dict

    # Aggregators
    def aggregate_table1(table1_dict):
        # sum matches/total for "All"
        for mk in table1_dict:
            sum_m = 0
            sum_t = 0
            mismatch_for_all = True
            for doc in DOC_NAMES:
                m, t, mm, _ = table1_dict[mk][doc]
                if not mm:
                    sum_m += m
                    sum_t += t
                    mismatch_for_all = False
            if mismatch_for_all:
                table1_dict[mk]["All"] = (0, 0, True, 0)
            else:
                table1_dict[mk]["All"] = (sum_m, sum_t, False, 0)

    def aggregate_table2(table2_dict):
        # table2_dict[mk][doc] => {col: (col_match, col_total)} or None
        # final => table2_final[mk][col] = (sum_m, sum_t)
        final = {}
        for mk in table2_dict:
            final[mk] = {}
            for c in EXPECTED_COLUMNS + ["All"]:
                final[mk][c] = [0, 0]
            for doc in DOC_NAMES:
                doc_dict = table2_dict[mk][doc]
                if doc_dict is None:
                    continue
                for c in EXPECTED_COLUMNS:
                    cm, ct = doc_dict[c]
                    final[mk][c][0] += cm
                    final[mk][c][1] += ct
                    final[mk]["All"][0] += cm
                    final[mk]["All"][1] += ct
        return final

    # do it
    aggregate_table1(table1_exact)
    aggregate_table1(table1_fuzzy)
    aggregate_table1(table1_lev)
    table2_exact_final = aggregate_table2(table2_exact)
    table2_fuzzy_final = aggregate_table2(table2_fuzzy)
    table2_lev_final = aggregate_table2(table2_lev)

    # -------------------------------------------------------------------
    # Build HTML
    # -------------------------------------------------------------------
    html_head = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Benchmark CSV Extraction</title>
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
<h1>CSV Extraction Benchmark (Exact + Fuzzy + Levenshtein)</h1>
"""

    html_body = []

    # Some general remarks
    html_body.append(f"""
<p><strong>General Remarks:</strong><br>
We first record the <em>original</em> ground-truth column count for each doc 
(before filtering out unwanted columns). Then we filter to these 4 columns:
{EXPECTED_COLUMNS}. Row mismatches => "dim mismatch". 
Empty, "null", or whitespace cells are normalized to "" for both exact & fuzzy checks.<br>
Fuzzy uses Jaro-Winkler with a threshold of {FUZZY_THRESHOLD} (≥ {FUZZY_THRESHOLD} => match).<br>
Levenshtein-based uses: two cells match if <em>distance ≤ number_of_words_in_GT</em>.<br>
</p>
""")

    # ========== Table-building Helpers ==========
    def build_table1_html(table1_dict, table_title):
        out = []
        out.append(f"<h2>{table_title}</h2>")
        out.append("<table><tr>")
        out.append("<th>Model</th>")
        for doc in DOC_NAMES:
            out.append(f"<th>{doc}</th>")
        out.append("<th>All</th>")
        out.append("</tr>")

        # Rows: discovered models
        sorted_mks = sorted(table1_dict.keys(), key=lambda x: x.lower())
        for mk in sorted_mks:
            out.append(f"<tr><td class='model-name'>{mk}</td>")
            for doc in DOC_NAMES + ["All"]:
                matches, total, mismatch, pred_nrows = table1_dict[mk][doc]
                if doc != "All":
                    # original col count, row count, etc.
                    oc = orig_gt_col_count[doc]  # original GT columns
                    gt_r = gt_row_counts[doc]    # GT row count
                else:
                    oc = sum(orig_gt_col_count[d] for d in DOC_NAMES)
                    gt_r = sum(gt_row_counts[d] for d in DOC_NAMES)

                if mismatch and doc != "All":
                    cell_val = f"<span class='mismatch'>dim mismatch (r={pred_nrows})</span>"
                elif mismatch and doc == "All":
                    cell_val = f"<span class='mismatch'>0.00% | 0</span>"
                else:
                    if total > 0:
                        rate = (matches / total) * 100.0
                        cell_val = f"{rate:.2f}% | {matches}"
                    else:
                        cell_val = "0.00% | 0"
                out.append(f"<td>{cell_val}</td>")
            out.append("</tr>")

        # 1) "Total number of cells": original GT column count * GT row count
        out.append("<tr><td class='model-name'>Total number of cells</td>")
        for doc in DOC_NAMES:
            r = gt_row_counts[doc]
            c = orig_gt_col_count[doc]
            out.append(f"<td>{r * c}</td>")
        sum_r = sum(gt_row_counts[d] for d in DOC_NAMES)
        sum_c = sum(orig_gt_col_count[d] for d in DOC_NAMES)
        out.append(f"<td>{sum_r * sum_c}</td>")
        out.append("</tr>")

        # 2) GT Rows
        out.append("<tr><td class='model-name'>GT Rows</td>")
        for doc in DOC_NAMES:
            out.append(f"<td>{gt_row_counts[doc]}</td>")
        out.append(f"<td>{sum_r}</td>")
        out.append("</tr>")

        out.append("</table>")
        return "\n".join(out)

    def build_table2_html(table2_final_dict, table_title):
        # table2_final_dict[mk][col] => (matches, total)
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

    # ========== EXACT Tables (1 & 2) ==========
    html_body.append(build_table1_html(table1_exact, "Table 1 (Exact): By Type"))
    html_body.append("""
<div class="note">
  <strong>Notes for Table 1 (Exact):</strong>
  <ul>
    <li>Cells are "XX.XX% | #hits" if dimension matches, otherwise "dim mismatch (r=NN)".</li>
    <li>"Total number of cells" = GT row count * original GT column count.</li>
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

    # ========== FUZZY Tables (3 & 4) ==========
    html_body.append(build_table1_html(table1_fuzzy, "Table 3 (Fuzzy): By Type"))
    html_body.append(f"""
<div class="note">
  <strong>Notes for Table 3 (Fuzzy):</strong>
  <ul>
    <li>A cell is considered a match if Jaro-Winkler similarity ≥ {FUZZY_THRESHOLD}.</li>
    <li>Dimension mismatch is the same concept: row counts must match.</li>
  </ul>
</div>
""")

    html_body.append(build_table2_html(table2_fuzzy_final, "Table 4 (Fuzzy): By Variable"))
    html_body.append(f"""
<div class="note">
  <strong>Notes for Table 4 (Fuzzy):</strong>
  <ul>
    <li>Mismatched docs are skipped, same logic as Table 2. Each cell shows "XX.XX% (#hits / #total)".</li>
    <li>Two cells match if their Jaro-Winkler similarity ≥ {FUZZY_THRESHOLD}.</li>
  </ul>
</div>
""")

    # ========== LEVENSHTEIN Tables (5 & 6) ==========
    html_body.append(build_table1_html(table1_lev, "Table 5 (Levenshtein): By Type"))
    html_body.append(f"""
<div class="note">
  <strong>Notes for Table 5 (Levenshtein):</strong>
  <ul>
    <li>A cell is considered a match if Levenshtein distance ≤ (number_of_words_in_GT_cell).</li>
    <li>Dimension mismatch is the same concept: row counts must match.</li>
    <li>Example: "hello how are we" (4 words) and "helloa ho aree wee" => distance ≤ 4 => match.</li>
  </ul>
</div>
""")

    html_body.append(build_table2_html(table2_lev_final, "Table 6 (Levenshtein): By Variable"))
    html_body.append(f"""
<div class="note">
  <strong>Notes for Table 6 (Levenshtein):</strong>
  <ul>
    <li>Mismatched docs are skipped, same logic as Table 2. Each cell shows "XX.XX% (#hits / #total)".</li>
    <li>Two cells match if Levenshtein distance ≤ (number_of_words_in_GT_cell).</li>
  </ul>
</div>
""")

    # Overall footer
    used_cores = args.n_jobs if args.n_jobs != -1 else os.cpu_count()
    html_body.append(f"""
<div class="section-note">
  <strong>Implementation Details:</strong><br>
  - Parallelized with joblib, n_jobs={used_cores}.<br>
  - For fuzzy matching, we used Jaro-Winkler via <code>rapidfuzz.distance.JaroWinkler.similarity()</code> 
    (≥ {FUZZY_THRESHOLD} => match).<br>
  - For Levenshtein-based, <em>distance ≤ number_of_words_in_ground_truth</em> => match.<br>
  <strong>Alternative Fuzzy Ideas:</strong> 
  We could use partial-ratio, token-set ratio, or other advanced methods from 
  <em>rapidfuzz.fuzz</em> to handle differences in word order or partial matches.
</div>
</body>
</html>
""")

    final_html = html_head + "\n".join(html_body)
    out_path = os.path.join(script_dir, args.output_html)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    logger.info("Done! HTML report (Exact + Fuzzy + Levenshtein) saved to '%s'.", out_path)


if __name__ == "__main__":
    main()