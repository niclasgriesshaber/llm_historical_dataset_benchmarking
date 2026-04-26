#!/usr/bin/env python3
"""
Runner for all revision experiments.

Runs all 8 pipelines × 10 document types = 80 experiments.
On failure: logs the error, skips that experiment, continues to the next.
At the end: prints a summary of all failures so the user can decide what to rerun.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# All experiments to run: (description, command)
EXPERIMENTS = []

# --- img2csv ---
for t in range(1, 11):
    EXPERIMENTS.append((
        f"img2csv/gemini-3.1/type-{t}",
        [sys.executable, str(PROJECT_ROOT / "src/llm_img2csv/gemini-3.1.py"),
         "--pdf", f"type-{t}.pdf", "--temperature", "0.0"]
    ))
    EXPERIMENTS.append((
        f"img2csv/gpt-5.5/type-{t}",
        [sys.executable, str(PROJECT_ROOT / "src/llm_img2csv/gpt-5.5.py"),
         "--pdf", f"type-{t}.pdf", "--temperature", "0.0"]
    ))

# --- img2txt ---
for t in range(1, 11):
    EXPERIMENTS.append((
        f"img2txt/gemini-3.1/type-{t}",
        [sys.executable, str(PROJECT_ROOT / "src/llm_img2txt/gemini-3.1.py"),
         "--pdf", f"type-{t}.pdf", "--temperature", "0.0"]
    ))
    EXPERIMENTS.append((
        f"img2txt/gpt-5.5/type-{t}",
        [sys.executable, str(PROJECT_ROOT / "src/llm_img2txt/gpt-5.5.py"),
         "--pdf", f"type-{t}.pdf", "--temperature", "0.0"]
    ))

# --- img2txt with Transkribus (post-OCR correction) ---
for t in range(1, 11):
    EXPERIMENTS.append((
        f"post-ocr/gemini-3.1/type-{t}",
        [sys.executable, str(PROJECT_ROOT / "src/llm_img2txt/gemini-3.1_with_transkribusOCR.py"),
         "--pdf", f"type-{t}.pdf", "--temperature", "0.0"]
    ))
    EXPERIMENTS.append((
        f"post-ocr/gpt-5.5/type-{t}",
        [sys.executable, str(PROJECT_ROOT / "src/llm_img2txt/gpt-5.5_with_transkribusOCR.py"),
         "--pdf", f"type-{t}.pdf", "--temperature", "0.0"]
    ))

# --- txt2csv ---
for t in range(1, 11):
    EXPERIMENTS.append((
        f"txt2csv/gemini-3.1/type-{t}",
        [sys.executable, str(PROJECT_ROOT / "src/llm_txt2csv/gemini-3.1.py"),
         "--txt", f"type-{t}.txt", "--temperature", "0.0"]
    ))
    EXPERIMENTS.append((
        f"txt2csv/gpt-5.5/type-{t}",
        [sys.executable, str(PROJECT_ROOT / "src/llm_txt2csv/gpt-5.5.py"),
         "--txt", f"type-{t}.txt", "--temperature", "0.0"]
    ))


def main():
    total = len(EXPERIMENTS)
    failures = []
    successes = 0
    overall_start = time.time()

    print(f"{'='*60}")
    print(f"REVISION EXPERIMENTS RUNNER")
    print(f"Total experiments: {total}")
    print(f"{'='*60}\n")

    for i, (name, cmd) in enumerate(EXPERIMENTS, 1):
        print(f"[{i}/{total}] {name}")
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=600  # 10 min max per experiment
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                # Script exited with error
                # Extract last few lines of stderr/stdout for the error message
                error_output = (result.stderr or result.stdout or "unknown error").strip()
                last_lines = "\n".join(error_output.split("\n")[-3:])
                print(f"  FAILED ({elapsed:.0f}s): {last_lines}\n")
                failures.append((name, last_lines))
            else:
                successes += 1
                print(f"  OK ({elapsed:.0f}s)\n")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>600s)\n")
            failures.append((name, "Timed out after 600 seconds"))
        except Exception as e:
            print(f"  ERROR: {e}\n")
            failures.append((name, str(e)))

    # Final summary
    total_time = time.time() - overall_start
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Succeeded: {successes}/{total}")
    print(f"Failed:    {len(failures)}/{total}")
    print(f"Total time: {hours}h {mins}m")

    if failures:
        print(f"\n{'='*60}")
        print(f"FAILURES (rerun these manually):")
        print(f"{'='*60}")
        for name, error in failures:
            print(f"\n  {name}")
            print(f"    Error: {error}")

    print()


if __name__ == "__main__":
    main()
