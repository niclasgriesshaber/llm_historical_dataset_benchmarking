#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
txt_accuracy.py

Generates a single HTML file ("txt_accuracy.html") containing side-by-side
character-level diffs of ground truth vs. generated text.

Key Details:
------------
  - Removes trailing whitespace and line breaks at the very end of each text
    (both ground truth and generated). No other text normalization is performed.
  - Uses RapidFuzz's Levenshtein.editops for character-level insert/delete/replace.
  - Produces one single HTML file for all (PDF, model) pairs.
  - Detects whether the model is LLM or OCR based on directory structure.
"""

import os
import argparse
from rapidfuzz.distance import Levenshtein

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side diff (character-level) vs. ground truth."
    )
    parser.add_argument(
        "--pdfs",
        type=str,
        nargs="+",
        required=True,
        help="One or more PDF filenames (e.g. type-1.pdf type-2.pdf)."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="One or more model names (e.g. gemini-2.0, gpt-4o, pytesseract, transkribus)."
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

def color_diff_char_level(gt: str, hyp: str):
    """
    Returns two HTML strings (colored_gt, colored_hyp) that highlight
    insertions, deletions, and substitutions at the character level,
    using <span> classes. Each character is processed exactly once.
    """
    ops = Levenshtein.editops(gt, hyp)
    colored_gt = []
    colored_hyp = []

    i_gt = 0
    i_hyp = 0

    for (op_type, src_i, dst_i) in ops:
        # Append any unchanged text prior to this edit
        if src_i > i_gt:
            colored_gt.append(gt[i_gt:src_i])
        if dst_i > i_hyp:
            colored_hyp.append(hyp[i_hyp:dst_i])

        if op_type == "insert":
            inserted_char = hyp[dst_i]
            colored_hyp.append(f'<span class="ins-span">{inserted_char}</span>')
            # Advance only in the hypothesis
            i_gt = src_i
            i_hyp = dst_i + 1

        elif op_type == "delete":
            deleted_char = gt[src_i]
            colored_gt.append(f'<span class="del-span">{deleted_char}</span>')
            # Advance only in the ground truth
            i_gt = src_i + 1
            i_hyp = dst_i

        elif op_type == "replace":
            replaced_char_gt = gt[src_i]
            replaced_char_hyp = hyp[dst_i]
            colored_gt.append(f'<span class="sub-span">{replaced_char_gt}</span>')
            colored_hyp.append(f'<span class="sub-span">{replaced_char_hyp}</span>')
            # Advance both
            i_gt = src_i + 1
            i_hyp = dst_i + 1

    # Append leftover text after the last edit op
    if i_gt < len(gt):
        colored_gt.append(gt[i_gt:])
    if i_hyp < len(hyp):
        colored_hyp.append(hyp[i_hyp:])

    return "".join(colored_gt), "".join(colored_hyp)

def add_line_numbers(diff_text: str) -> str:
    """
    Splits the diff text by newline, prepends line numbers,
    and rejoins with newlines.
    """
    lines = diff_text.split("\n")
    numbered_lines = []
    for i, line in enumerate(lines, start=1):
        numbered_lines.append(f"{i:4d} | {line}")
    return "\n".join(numbered_lines)

def is_llm_model(model_name: str, project_root: str) -> bool:
    """
    Checks if 'model_name' is under results/llm_img2txt/<model_name>.
    If so => LLM. Otherwise => OCR.
    """
    llm_path = os.path.join(project_root, "results", "llm_img2txt", model_name)
    return os.path.isdir(llm_path)

def main():
    args = parse_arguments()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    ground_truth_root = os.path.join(project_root, "data", "ground_truth", "txt")
    llm_root = os.path.join(project_root, "results", "llm_img2txt")
    ocr_root = os.path.join(project_root, "results", "ocr_img2txt")

    # Collect HTML chunks from each comparison
    all_comparisons_html = []

    for pdf_name in args.pdfs:
        doc_name = os.path.splitext(pdf_name)[0]
        gt_path = os.path.join(ground_truth_root, f"{doc_name}.txt")

        if not os.path.isfile(gt_path):
            print(f"[WARNING] No ground-truth for {doc_name}.txt => skipping.")
            continue

        # Read and strip trailing whitespace from ground truth
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read()
        gt_text = gt_text.rstrip()

        for model_name in args.models:
            if is_llm_model(model_name, project_root):
                gen_path = os.path.join(
                    llm_root,
                    model_name,
                    doc_name,
                    f"temperature_{args.temperature}",
                    args.run,
                    f"{doc_name}.txt",
                )
            else:
                gen_path = os.path.join(
                    ocr_root,
                    model_name,
                    doc_name,
                    args.run,
                    f"{doc_name}.txt",
                )

            if not os.path.isfile(gen_path):
                print(f"[WARNING] No generated text at {gen_path} => skipping.")
                continue

            # Read and strip trailing whitespace from generated
            with open(gen_path, "r", encoding="utf-8") as f:
                gen_text = f.read()
            gen_text = gen_text.rstrip()

            # Compute character-level distance & CER
            dist_char = Levenshtein.distance(gt_text, gen_text)
            gt_len = len(gt_text)
            cer = dist_char / gt_len if gt_len > 0 else 0.0

            # Build color-coded diffs, plus line numbering
            colored_gt, colored_gen = color_diff_char_level(gt_text, gen_text)
            colored_gt_numbered = add_line_numbers(colored_gt)
            colored_gen_numbered = add_line_numbers(colored_gen)

            # Build HTML snippet for this comparison
            comparison_html = f"""
<div class="container">
  <h1>Comparison: {doc_name}.pdf — Model: {model_name}</h1>
  <div class="table-scroll">
    <table class="diff-table">
      <tr>
        <th>Ground Truth (Line #)</th>
        <th>Generated (Line #)</th>
      </tr>
      <tr>
        <td><pre>{colored_gt_numbered}</pre></td>
        <td><pre>{colored_gen_numbered}</pre></td>
      </tr>
    </table>
  </div>
  <div class="summary">
    <strong>Levenshtein Distance:</strong> {dist_char} &nbsp;|&nbsp;
    <strong>Ground Truth Length:</strong> {gt_len} &nbsp;|&nbsp;
    <strong>CER:</strong> {cer:.2%}
  </div>
</div>
"""
            all_comparisons_html.append(comparison_html)
            print(f"[INFO] Processed => PDF: {pdf_name}, Model: {model_name}")

    # Write everything to a single file named txt_accuracy.html
    out_html_path = os.path.join(script_dir, "txt_accuracy.html")

    # Legend shown once at the end
    legend_html = """
<div class="legend">
  <ul>
    <li><span class="del-span">Deletion</span> (in ground truth)</li>
    <li><span class="ins-span">Insertion</span> (in generated)</li>
    <li><span class="sub-span">Substitution</span> (both sides)</li>
  </ul>
</div>
"""

    final_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Text Accuracy Comparisons</title>
<style>
body {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  margin: 20px;
  background: #fafafa;
  color: #333;
}}
h1 {{
  text-align: center;
  font-size: 1.8em;
  margin-bottom: 0.3em;
}}
.container {{
  display: flex;
  flex-direction: column;
  gap: 1em;
  margin-bottom: 2em;
}}
.table-scroll {{
  max-height: 600px;
  overflow: auto;
  border: 1px solid #ddd;
}}
.diff-table {{
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed; /* consistent column widths */
}}
.diff-table th {{
  background: #4CAF50;
  color: white;
  text-align: center;
  padding: 8px;
}}
.diff-table td {{
  width: 50%;
  vertical-align: top;
  border: 1px solid #ddd;
  padding: 5px;
}}
/* Force wrapping in <pre> so long lines don't overflow columns */
.diff-table pre {{
  margin: 0;
  padding: 10px;
  font-family: "Courier New", monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
  overflow-wrap: break-word;
}}
.del-span {{
  background-color: #ffecec; /* pink/red for deleted chars */
  color: #d00;
}}
.ins-span {{
  background-color: #ecffec; /* green for inserted chars */
  color: #080;
}}
.sub-span {{
  background-color: #fff3cd; /* light orange for substituted chars */
  color: #b45f06;
}}
.summary {{
  text-align: center;
  margin: 0.5em 0 0.5em 0;
  font-size: 1.1em;
}}
.legend {{
  margin-top: 1.5em;
  font-size: 1em;
  color: #555;
  text-align: center;
}}
.legend ul {{
  list-style: none;
  padding-left: 0;
  margin: 0 auto;
  max-width: 400px;
}}
.legend li {{
  margin: 0.25em 0;
}}
.legend .del-span,
.legend .ins-span,
.legend .sub-span {{
  font-weight: bold;
  padding: 0 0.2em;
}}
</style>
</head>
<body>

{"".join(all_comparisons_html)}

{legend_html}

</body>
</html>
"""

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"[DONE] All comparisons combined into => {out_html_path}")

if __name__ == "__main__":
    main()