#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
csv_accuracy.py

Generates a single HTML file ("csv_accuracy.html") comparing ground truth CSV
rows/cells vs. generated CSV rows/cells, for multiple (CSV, model) pairs.

Directory structure specifics:
------------------------------
  - Ground truth CSVs are located at:
       data/ground_truth/csv/<doc_name_no_ext>.csv

  - Generated CSVs have this pattern:
       results/llm_img2csv/<model>/<doc_name_no_ext>.pdf/temperature_<temp>/<run>/<doc_name_no_ext>.csv
       results/llm_pdf2csv/<model>/<doc_name_no_ext>.pdf/temperature_<temp>/<run>/<doc_name_no_ext>.csv
       results/llm_txt2csv/<model>/<doc_name_no_ext>/temperature_<temp>/<run>/<doc_name_no_ext>.csv
    where, for example, if the input CSV is "type-1.csv", then <doc_name_no_ext> is "type-1",
    but for img/pdf we use a folder named "type-1.pdf", and for txt we use just "type-1".

  - CSVs all have these 5 columns (exact strings):
       [ "first and middle names", "surname", "occupation", "address", "id" ]

    Any extra columns in generated CSV (like "page_number") are ignored,
    as long as the 5 required columns are present.

  - For the first 4 columns:
       * Matching cells => green
       * Non-matching => red with the format: <generated_val> / <ground_truth_val> (gt)

    The "id" column is shown but never colored.

  - One big HTML file is produced, with sections for each (CSV, model, subtask).
"""

import argparse
import os
import csv

# The required columns (as specified) in correct order:
REQUIRED_COLUMNS = [
    "first and middle names",
    "surname",
    "occupation",
    "address",
    "id",
]

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side comparisons of ground truth vs. generated CSV."
    )
    parser.add_argument(
        "--csvs",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV filenames (e.g. type-1.csv type-2.csv)."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="One or more model names (e.g. gemini-2.0, gpt-4o)."
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0.0",
        help="Temperature folder for LLM results (default: 0.0)."
    )
    parser.add_argument(
        "--run",
        type=str,
        default="run_01",
        help="Run folder name (default: run_01)."
    )
    return parser.parse_args()

def read_csv_file(csv_path):
    """
    Reads a CSV (with header) into a list of rows (dicts).
    Returns (header_list, row_dicts_list).

    If file not found, returns (None, []).
    """
    if not os.path.isfile(csv_path):
        return None, []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    return header, rows

def filter_columns(header, rows):
    """
    Given a CSV header and list of row dicts, remove any columns
    that are NOT in the REQUIRED_COLUMNS set.
    Returns (filtered_header, filtered_rows).
    """
    required_set = set(REQUIRED_COLUMNS)

    # Filter the header
    filtered_header = [col for col in header if col in required_set]

    # Filter each row
    filtered_rows = []
    for row in rows:
        new_row = {}
        for col in filtered_header:
            # Copy only required columns that exist in this row
            new_row[col] = row.get(col, "")
        filtered_rows.append(new_row)

    return filtered_header, filtered_rows

def validate_and_reorder(
    header_gt, rows_gt,
    header_gen, rows_gen
):
    """
    1. Filter out extra columns in both ground truth and generated.
    2. Check if each still has all REQUIRED_COLUMNS after filtering.
    3. Check row counts match.
    4. Reorder generated columns to match the (filtered) ground-truth column order.
    5. Return (False, reason) or (True, (final_header_gt, final_rows_gt, reordered_rows_gen)).
    """

    # Filter out any extra columns from ground truth
    header_gt_f, rows_gt_f = filter_columns(header_gt, rows_gt)
    # Filter out any extra columns from generated
    header_gen_f, rows_gen_f = filter_columns(header_gen, rows_gen)

    # Now check if each has the full set of required columns
    required_set = set(REQUIRED_COLUMNS)
    if set(header_gt_f) != required_set:
        return False, "Ground truth missing one or more required columns"
    if set(header_gen_f) != required_set:
        return False, "Generated CSV missing one or more required columns"

    # Next, check row counts
    if len(rows_gt_f) != len(rows_gen_f):
        return False, "Row counts differ"

    # Reorder generated columns to match ground truth's filtered order
    # (The ground truth's filtered header might be a permutation of REQUIRED_COLUMNS.)
    reordered_rows_gen = []
    for row in rows_gen_f:
        new_row = {}
        for col in header_gt_f:
            new_row[col] = row[col]
        reordered_rows_gen.append(new_row)

    # Return the newly filtered ground-truth + reordered generated
    return True, (header_gt_f, rows_gt_f, reordered_rows_gen)

def generate_html_table(
    doc_name_no_ext, model_name, subtask_label,
    final_header_gt, final_rows_gt,
    reordered_rows_gen
):
    """
    Build an HTML snippet for the table, comparing row by row, cell by cell.
    final_header_gt is the filtered header for ground truth (in the correct order).
    final_rows_gt is the filtered rows for ground truth.
    reordered_rows_gen is the generated rows with columns in the same order.

    For the first 4 columns, match => green, mismatch => red ("gen / gt (gt)").
    The 'id' column => no color.
    """

    html_snippets = []
    csv_display_name = doc_name_no_ext + ".csv"

    # Title
    html_snippets.append(
        f'<h1>Comparison: {csv_display_name} — Model: {model_name} ({subtask_label})</h1>'
    )
    html_snippets.append('<div class="table-container">')
    html_snippets.append('<table class="csv-table">')

    # Header row
    html_snippets.append('  <thead>')
    html_snippets.append('    <tr>')
    for col in final_header_gt:
        html_snippets.append(f'      <th>{col}</th>')
    html_snippets.append('    </tr>')
    html_snippets.append('  </thead>')

    # Body
    html_snippets.append('  <tbody>')
    for i, gt_row in enumerate(final_rows_gt):
        gen_row = reordered_rows_gen[i]
        html_snippets.append('    <tr>')
        for col in final_header_gt:
            gt_val = gt_row[col]
            gen_val = gen_row[col]

            # 'id' => no color
            if col == "id":
                html_snippets.append(f'      <td>{gt_val}</td>')
                continue

            # Check match
            if gt_val == gen_val:
                # green
                html_snippets.append(f'      <td class="cell-match">{gt_val}</td>')
            else:
                # red
                html_snippets.append(
                    f'      <td class="cell-mismatch">{gen_val} / {gt_val} (gt)</td>'
                )
        html_snippets.append('    </tr>')
    html_snippets.append('  </tbody>')

    html_snippets.append('</table>')
    html_snippets.append('</div>')  # table-container

    return "\n".join(html_snippets)

def main():
    args = parse_arguments()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    ground_truth_root = os.path.join(project_root, "data", "ground_truth", "csv")

    # Now also include txt2csv in our tasks:
    tasks = [
        ("img2csv", os.path.join(project_root, "results", "llm_img2csv")),
        ("pdf2csv", os.path.join(project_root, "results", "llm_pdf2csv")),
        ("txt2csv", os.path.join(project_root, "results", "llm_txt2csv")),
    ]

    all_comparisons_html = []

    for csv_name in args.csvs:
        # doc_name_no_ext = "type-1" if csv_name = "type-1.csv"
        doc_name_no_ext = os.path.splitext(csv_name)[0]

        gt_path = os.path.join(ground_truth_root, f"{doc_name_no_ext}.csv")
        if not os.path.isfile(gt_path):
            print(f"[WARNING] No ground-truth CSV for {doc_name_no_ext}.csv => skipping.")
            continue

        # Read ground truth
        header_gt, rows_gt = read_csv_file(gt_path)
        if header_gt is None:
            print(f"[WARNING] Failed to read ground truth CSV: {gt_path}")
            continue

        for model_name in args.models:
            # Track whether any CSV was found for this model
            found_any_csv_for_model = False

            for subtask_label, subtask_root in tasks:
                # All subtasks use the same folder name
                folder_name = doc_name_no_ext

                # Generate the path to the CSV file
                gen_path = os.path.join(
                    subtask_root,
                    model_name,
                    folder_name,
                    f"temperature_{args.temperature}",
                    args.run,
                    f"{doc_name_no_ext}.csv"
                )

                if not os.path.isfile(gen_path):
                    # No file => skip
                    continue

                found_any_csv_for_model = True
                header_gen, rows_gen = read_csv_file(gen_path)
                if header_gen is None:
                    print(f"[WARNING] Failed to read generated CSV: {gen_path}")
                    continue

                # Validate columns & rows (ignoring extras)
                valid, result = validate_and_reorder(
                    header_gt, rows_gt, header_gen, rows_gen
                )
                if not valid:
                    # Dimensions mismatch => short note
                    mismatch_html = f"""
<h1>Comparison: {doc_name_no_ext}.csv — Model: {model_name} ({subtask_label})</h1>
<p class="error">
  Dimensions do not match for {doc_name_no_ext}.csv — Model {model_name} ({subtask_label}). 
  Reason: {result}. No table produced.
</p>
"""
                    all_comparisons_html.append(mismatch_html)
                    print(f"[INFO] Mismatch => {csv_name}, Model: {model_name} ({subtask_label}). Reason: {result}")
                else:
                    final_header_gt, final_rows_gt, reordered_rows_gen = result
                    comparison_html = generate_html_table(
                        doc_name_no_ext, model_name, subtask_label,
                        final_header_gt, final_rows_gt,
                        reordered_rows_gen
                    )
                    all_comparisons_html.append(comparison_html)
                    print(f"[INFO] Processed => CSV: {csv_name}, Model: {model_name} ({subtask_label})")

            # If no CSV found in any subtask, note that
            if not found_any_csv_for_model:
                no_file_html = f"""
<h1>Comparison: {doc_name_no_ext}.csv — Model: {model_name}</h1>
<p class="error">
  No generated CSV found in img2csv, pdf2csv, or txt2csv for {doc_name_no_ext}.csv (Model: {model_name}).
</p>
"""
                all_comparisons_html.append(no_file_html)
                print(f"[INFO] No generated CSV => {csv_name}, Model: {model_name}")

    # Write the final HTML
    out_html_path = os.path.join(script_dir, "csv_accuracy.html")

    style_block = """
<style>
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  background: #fafafa;
  color: #333;
  margin: 20px;
}
h1 {
  font-size: 1.4em;
  margin-bottom: 0.3em;
}
.table-container {
  overflow-x: auto;
  margin-bottom: 2em;
}
.csv-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1em;
}
.csv-table th {
  background: #007BFF;
  color: #fff;
  padding: 8px;
  text-align: left;
  border: 1px solid #ccc;
  white-space: nowrap;
}
.csv-table td {
  border: 1px solid #ccc;
  padding: 8px;
  vertical-align: top;
  white-space: nowrap;
}
.cell-match {
  background-color: #c6f6d5; /* light green */
}
.cell-mismatch {
  background-color: #fed7d7; /* light red */
}
.error {
  color: #b00;
  font-weight: bold;
  margin-bottom: 1em;
}
</style>
"""

    final_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>CSV Accuracy Comparisons</title>
{style_block}
</head>
<body>

{"".join(all_comparisons_html)}

</body>
</html>
"""

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"[DONE] All CSV comparisons combined into => {out_html_path}")

if __name__ == "__main__":
    main()