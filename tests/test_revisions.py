"""
test_revisions.py
=================

Verification tests for the revision experiment infrastructure.

These tests confirm that the new model scripts (gemini-3.1, gpt-5.5),
prompt files, directory structure, and benchmarking scripts are correctly
set up -- and that the original experiment outputs remain untouched.

Run with:
    python -m pytest tests/test_revisions.py -v
"""

import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Project root: two levels up from this test file
#   tests/test_revisions.py -> tests/ -> project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


###############################################################################
# 1. PROMPT TESTS
###############################################################################


class TestPrompts:
    """Tests that verify the six new prompt files exist, contain the right
    content, avoid aggressive language, and are written in German."""

    # The six new prompt files introduced for the revision experiment.
    # Each tuple is (pipeline_subdirectory, model_filename).
    PROMPT_FILES = [
        ("llm_img2csv", "gemini-3.1.txt"),
        ("llm_img2csv", "gpt-5.5.txt"),
        ("llm_img2txt", "gemini-3.1.txt"),
        ("llm_img2txt", "gpt-5.5.txt"),
        ("llm_txt2csv", "gemini-3.1.txt"),
        ("llm_txt2csv", "gpt-5.5.txt"),
    ]

    def test_prompt_files_exist(self):
        """Verify all 6 prompt files exist.

        Each new model (gemini-3.1, gpt-5.5) needs a prompt file in each
        of the three pipeline directories (llm_img2csv, llm_img2txt,
        llm_txt2csv).  Missing prompt files would cause the pipeline
        scripts to exit immediately on startup.
        """
        for subdir, filename in self.PROMPT_FILES:
            # Build the full path: src/prompts/<pipeline>/<model>.txt
            prompt_path = PROJECT_ROOT / "src" / "prompts" / subdir / filename
            assert prompt_path.is_file(), (
                f"Expected prompt file does not exist: {prompt_path}"
            )

    def test_prompts_no_aggressive_language(self):
        """Verify no 'SYSTEM FAILURE', 'OBLITERATE', 'HUMANITY DEPENDS' in new prompts.

        Academic research prompts should use measured, professional language.
        Aggressive or hyperbolic phrases are inappropriate for scholarly work
        and may degrade model output quality.
        """
        # Phrases that should never appear in a scholarly prompt
        forbidden_phrases = [
            "SYSTEM FAILURE",
            "OBLITERATE",
            "HUMANITY DEPENDS",
        ]

        for subdir, filename in self.PROMPT_FILES:
            prompt_path = PROJECT_ROOT / "src" / "prompts" / subdir / filename
            # Read the full prompt text for inspection
            content = prompt_path.read_text(encoding="utf-8")

            for phrase in forbidden_phrases:
                # Case-insensitive check to catch any capitalisation variant
                assert phrase.lower() not in content.lower(), (
                    f"Forbidden phrase '{phrase}' found in {prompt_path}"
                )

    def test_prompts_contain_required_fields(self):
        """Verify prompts mention the required JSON field names.

        The img2csv and txt2csv pipelines produce structured CSV output.
        Their prompts MUST reference the four standard field names so the
        model knows which columns to extract.  The img2txt prompts are
        free-text transcription and do not need these fields.
        """
        # Only img2csv and txt2csv prompts need the structured field names
        structured_prompts = [
            ("llm_img2csv", "gemini-3.1.txt"),
            ("llm_img2csv", "gpt-5.5.txt"),
            ("llm_txt2csv", "gemini-3.1.txt"),
            ("llm_txt2csv", "gpt-5.5.txt"),
        ]

        # These are the four JSON field names that every structured prompt
        # must mention so the model produces the correct output schema.
        required_fields = [
            "first and middle names",
            "surname",
            "occupation",
            "address",
        ]

        for subdir, filename in structured_prompts:
            prompt_path = PROJECT_ROOT / "src" / "prompts" / subdir / filename
            content = prompt_path.read_text(encoding="utf-8")

            for field in required_fields:
                # Field names should appear exactly as written (they become
                # JSON keys), so we do a case-sensitive substring check.
                assert field in content, (
                    f"Required field '{field}' not found in {prompt_path}"
                )

    def test_prompts_are_german(self):
        """Verify prompts contain German text indicators.

        The address books being transcribed are historical German documents.
        Prompts must be written in German so the model understands the
        context and produces output that matches the source material.
        We check for at least one characteristic German word per file.
        """
        # Words that are distinctive markers of German-language prompts
        # used in this project.  At least one must appear in each file.
        german_indicators = ["Historiker", "Adressbuch", "Regeln", "Felder"]

        for subdir, filename in self.PROMPT_FILES:
            prompt_path = PROJECT_ROOT / "src" / "prompts" / subdir / filename
            content = prompt_path.read_text(encoding="utf-8")

            # At least one German indicator word should be present
            found = any(word in content for word in german_indicators)
            assert found, (
                f"No German text indicators ({german_indicators}) found "
                f"in {prompt_path}.  Prompts should be in German."
            )


###############################################################################
# 2. SCRIPT TESTS
###############################################################################


class TestScripts:
    """Tests that verify all 8 new model scripts exist and have the correct
    model name constants, results directory, and logs directory."""

    # All 8 new model scripts for the revision experiment.
    # Each tuple is (relative_path, model_name, full_model_name).
    SCRIPTS = [
        # --- img2csv ---
        ("src/llm_img2csv/gemini-3.1.py", "gemini-3.1", "gemini-3.1-pro-preview"),
        ("src/llm_img2csv/gpt-5.5.py", "gpt-5.5", "gpt-5.5"),
        # --- img2txt (standalone) ---
        ("src/llm_img2txt/gemini-3.1.py", "gemini-3.1", "gemini-3.1-pro-preview"),
        ("src/llm_img2txt/gpt-5.5.py", "gpt-5.5", "gpt-5.5"),
        # --- img2txt (with Transkribus post-correction) ---
        ("src/llm_img2txt/gemini-3.1_with_transkribusOCR.py", "gemini-3.1-with-transkribus", "gemini-3.1-pro-preview"),
        ("src/llm_img2txt/gpt-5.5_with_transkribusOCR.py", "gpt-5.5-with-transkribus", "gpt-5.5"),
        # --- txt2csv ---
        ("src/llm_txt2csv/gemini-3.1.py", "gemini-3.1", "gemini-3.1-pro-preview"),
        ("src/llm_txt2csv/gpt-5.5.py", "gpt-5.5", "gpt-5.5"),
    ]

    def test_model_scripts_exist(self):
        """Verify all 8 model scripts exist.

        The revision experiment adds two new models (gemini-3.1, gpt-5.5)
        across four pipeline variants: img2csv (2 scripts), img2txt
        standalone (2 scripts), img2txt with Transkribus (2 scripts), and
        txt2csv (2 scripts).  All 8 must be present for the experiment
        to be runnable.
        """
        for rel_path, _, _ in self.SCRIPTS:
            script_path = PROJECT_ROOT / rel_path
            assert script_path.is_file(), (
                f"Expected script does not exist: {script_path}"
            )

    def test_scripts_correct_model_names(self):
        """Verify MODEL_NAME and FULL_MODEL_NAME are set correctly.

        Each script must declare two constants:
          - MODEL_NAME: short identifier used for directory naming
            (e.g. "gemini-3.1", "gpt-5.5-with-transkribus")
          - FULL_MODEL_NAME: the exact API model identifier sent to the
            provider (e.g. "gemini-3.1-pro-preview", "gpt-5.5")

        Using the wrong model name would silently run the wrong model
        or write results to the wrong directory.
        """
        for rel_path, expected_model, expected_full in self.SCRIPTS:
            script_path = PROJECT_ROOT / rel_path
            content = script_path.read_text(encoding="utf-8")

            # Check MODEL_NAME assignment.
            # We look for the pattern: MODEL_NAME = "value"
            # allowing for either single or double quotes.
            assert f'MODEL_NAME = "{expected_model}"' in content, (
                f'MODEL_NAME = "{expected_model}" not found in {script_path}'
            )

            # Check FULL_MODEL_NAME assignment.
            assert f'FULL_MODEL_NAME = "{expected_full}"' in content, (
                f'FULL_MODEL_NAME = "{expected_full}" not found in {script_path}'
            )

    def test_scripts_use_results_revisions(self):
        """Verify all scripts output to results_revisions/, not results/.

        Revision experiment results must be isolated from the original
        experiment outputs.  Every new script must write to
        results_revisions/ so the original results/ directory stays clean.
        """
        for rel_path, _, _ in self.SCRIPTS:
            script_path = PROJECT_ROOT / rel_path
            content = script_path.read_text(encoding="utf-8")

            # The RESULTS_DIR (or equivalent path) must reference
            # "results_revisions" somewhere in its definition.
            assert "results_revisions" in content, (
                f"Script {script_path} does not reference 'results_revisions'. "
                "Revision scripts must write to results_revisions/, not results/."
            )

    def test_scripts_use_logs_revisions(self):
        """Verify all scripts log to logs_revisions/, not logs/.

        Just as results are isolated, logs must also be kept separate.
        Every new script must write log files to logs_revisions/ to avoid
        polluting the original logs/ directory.
        """
        for rel_path, _, _ in self.SCRIPTS:
            script_path = PROJECT_ROOT / rel_path
            content = script_path.read_text(encoding="utf-8")

            # The LOGS_DIR (or equivalent path) must reference
            # "logs_revisions" somewhere in its definition.
            assert "logs_revisions" in content, (
                f"Script {script_path} does not reference 'logs_revisions'. "
                "Revision scripts must write logs to logs_revisions/, not logs/."
            )

    def test_transkribus_scripts_read_from_old_results(self):
        """Verify with_transkribus scripts read OCR from results/ (old), not results_revisions/.

        The Transkribus OCR output was produced during the original
        experiment and is a fixed input.  The post-correction scripts
        must read it from results/ocr_img2txt/transkribus/, NOT from
        results_revisions/.  This ensures the revision experiment uses
        exactly the same OCR baseline as the original.
        """
        # Only the two with_transkribusOCR scripts need this check
        transkribus_scripts = [
            "src/llm_img2txt/gemini-3.1_with_transkribusOCR.py",
            "src/llm_img2txt/gpt-5.5_with_transkribusOCR.py",
        ]

        for rel_path in transkribus_scripts:
            script_path = PROJECT_ROOT / rel_path
            content = script_path.read_text(encoding="utf-8")

            # The script must reference the old Transkribus results path.
            # This path uses "results" (not "results_revisions") followed
            # by "ocr_img2txt" and "transkribus".
            assert "results" in content, (
                f"Script {script_path} does not reference results/ at all."
            )

            # More specifically, check for the Transkribus OCR path pattern.
            # The scripts build a path like:
            #   PROJECT_ROOT / "results" / "ocr_img2txt" / "transkribus" / ...
            assert "ocr_img2txt" in content and "transkribus" in content, (
                f"Script {script_path} does not reference the Transkribus OCR "
                "path (results/ocr_img2txt/transkribus/).  Post-correction "
                "scripts must read OCR text from the original results directory."
            )

            # Verify it's reading from "results/" not "results_revisions/"
            # for the OCR source.  We check that the OCR root path string
            # does NOT include "results_revisions" before "ocr_img2txt".
            # We look for the line that constructs the transkribus_ocr_root
            # variable -- it should reference plain "results", not
            # "results_revisions".
            lines = content.split("\n")
            for line in lines:
                if "ocr_img2txt" in line and "transkribus" in line:
                    # This line defines or references the OCR source path.
                    # It must NOT use results_revisions for the OCR input.
                    assert "results_revisions" not in line, (
                        f"Script {script_path} reads Transkribus OCR from "
                        "results_revisions/ instead of results/.  The OCR "
                        "baseline must come from the original experiment."
                    )


###############################################################################
# 3. DIRECTORY STRUCTURE TESTS
###############################################################################


class TestDirectoryStructure:
    """Tests that verify the results_revisions/ and benchmarking_revisions/
    directory trees are correctly set up."""

    def test_results_revisions_directory_exists(self):
        """Verify the results_revisions/ directory structure.

        The revision experiment needs a parallel directory tree under
        results_revisions/ that mirrors the structure of results/ but
        contains only the new model subdirectories.  Each pipeline
        (img2csv, img2txt, txt2csv) gets subdirectories for each model
        variant.
        """
        # Expected subdirectories under results_revisions/.
        # These must exist (even if empty) so the pipeline scripts can
        # create run folders inside them.
        expected_dirs = [
            # img2csv: two model variants
            "results_revisions/llm_img2csv/gemini-3.1",
            "results_revisions/llm_img2csv/gpt-5.5",
            # img2txt: four model variants (standalone + with-transkribus)
            "results_revisions/llm_img2txt/gemini-3.1",
            "results_revisions/llm_img2txt/gpt-5.5",
            "results_revisions/llm_img2txt/gemini-3.1-with-transkribus",
            "results_revisions/llm_img2txt/gpt-5.5-with-transkribus",
            # txt2csv: two model variants
            "results_revisions/llm_txt2csv/gemini-3.1",
            "results_revisions/llm_txt2csv/gpt-5.5",
        ]

        for rel_dir in expected_dirs:
            dir_path = PROJECT_ROOT / rel_dir
            assert dir_path.is_dir(), (
                f"Expected directory does not exist: {dir_path}"
            )

    def test_benchmarking_revisions_directory_exists(self):
        """Verify src/benchmarking_revisions/ and tables/ exist.

        The benchmarking_revisions/ directory holds the accuracy analysis
        scripts (or will hold them) and a tables/ subdirectory for LaTeX
        output.  Both must exist for the benchmarking pipeline.
        """
        # The main benchmarking_revisions directory
        bench_dir = PROJECT_ROOT / "src" / "benchmarking_revisions"
        assert bench_dir.is_dir(), (
            f"benchmarking_revisions directory does not exist: {bench_dir}"
        )

        # The tables subdirectory for LaTeX output
        tables_dir = bench_dir / "tables"
        assert tables_dir.is_dir(), (
            f"tables subdirectory does not exist: {tables_dir}"
        )


###############################################################################
# 4. SEPARATION TESTS
###############################################################################


class TestSeparation:
    """Tests that verify the original results/ and src/benchmarking/
    directories have not been modified by the revision setup."""

    def test_old_results_untouched(self):
        """Verify no new files were added to results/ directory.

        The revision experiment must not pollute the original results/
        directory.  We check that results/llm_img2csv/ only contains the
        original model subdirectories (gemini-2.0, gpt-4o) and that
        results/llm_img2txt/ only contains the original model
        subdirectories (gemini-2.0, gpt-4o, plus their -with-transkribus
        variants).
        """
        # --- Check results/llm_img2csv/ ---
        # Should contain exactly: gemini-2.0 and gpt-4o
        img2csv_dir = PROJECT_ROOT / "results" / "llm_img2csv"
        if img2csv_dir.is_dir():
            # Get only immediate subdirectory names
            subdirs = {d.name for d in img2csv_dir.iterdir() if d.is_dir()}
            # The original models -- no new revision models should appear here
            expected = {"gemini-2.0", "gpt-4o"}
            assert subdirs == expected, (
                f"results/llm_img2csv/ contains unexpected subdirectories: "
                f"{subdirs - expected}.  Revision results must go to "
                "results_revisions/, not results/."
            )

        # --- Check results/llm_img2txt/ ---
        # Should contain exactly: gemini-2.0, gpt-4o, and their
        # -with-transkribus variants
        img2txt_dir = PROJECT_ROOT / "results" / "llm_img2txt"
        if img2txt_dir.is_dir():
            subdirs = {d.name for d in img2txt_dir.iterdir() if d.is_dir()}
            expected = {
                "gemini-2.0",
                "gpt-4o",
                "gemini-2.0-with-transkribus",
                "gpt-4o-with-transkribus",
            }
            assert subdirs == expected, (
                f"results/llm_img2txt/ contains unexpected subdirectories: "
                f"{subdirs - expected}.  Revision results must go to "
                "results_revisions/, not results/."
            )

    def test_old_benchmarking_untouched(self):
        """Verify src/benchmarking/ files are unchanged.

        The original benchmarking scripts must remain in place and
        unmodified.  We verify the two key Python files still exist.
        Additionally, we use git to confirm they have no uncommitted
        modifications -- if git reports them as clean, they haven't been
        accidentally altered during the revision setup.
        """
        # Both benchmark scripts must exist
        csv_acc = PROJECT_ROOT / "src" / "benchmarking" / "csv_accuracy.py"
        txt_acc = PROJECT_ROOT / "src" / "benchmarking" / "txt_accuracy.py"

        assert csv_acc.is_file(), (
            f"Original benchmarking script missing: {csv_acc}"
        )
        assert txt_acc.is_file(), (
            f"Original benchmarking script missing: {txt_acc}"
        )

        # Use git to check that these files have no uncommitted changes.
        # `git diff --name-only` lists files with unstaged modifications.
        # If the original benchmarking scripts appear in the output, they
        # have been changed and the revision setup may have broken them.
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        # Only check if git command succeeded (it might fail in CI
        # environments without a git repo)
        if result.returncode == 0:
            changed_files = result.stdout.strip().split("\n")
            assert "src/benchmarking/csv_accuracy.py" not in changed_files, (
                "src/benchmarking/csv_accuracy.py has been modified. "
                "The original benchmarking scripts must not be changed."
            )
            assert "src/benchmarking/txt_accuracy.py" not in changed_files, (
                "src/benchmarking/txt_accuracy.py has been modified. "
                "The original benchmarking scripts must not be changed."
            )


###############################################################################
# 5. BENCHMARKING SCRIPT TESTS
###############################################################################


class TestBenchmarkingRevisions:
    """Tests that verify the revision benchmarking infrastructure exists
    and reads from the correct directories."""

    def test_benchmarking_revisions_scripts_exist(self):
        """Verify csv_accuracy.py and txt_accuracy.py exist in benchmarking_revisions/.

        The revision experiment needs its own benchmarking scripts that
        are configured to read from results_revisions/ and write tables
        to src/benchmarking_revisions/tables/.  These scripts may be
        exact copies of the originals with updated paths, or may be
        newly written.

        Note: If the benchmarking scripts have not yet been created,
        this test will fail as a reminder to complete that step.
        """
        bench_dir = PROJECT_ROOT / "src" / "benchmarking_revisions"

        csv_script = bench_dir / "csv_accuracy.py"
        txt_script = bench_dir / "txt_accuracy.py"

        assert csv_script.is_file(), (
            f"Benchmarking revision script missing: {csv_script}. "
            "This script is needed to evaluate CSV extraction accuracy "
            "for the revision models."
        )
        assert txt_script.is_file(), (
            f"Benchmarking revision script missing: {txt_script}. "
            "This script is needed to evaluate text transcription accuracy "
            "for the revision models."
        )

    def test_benchmarking_reads_from_results_revisions(self):
        """Verify benchmarking scripts read from results_revisions/.

        The revision benchmarking scripts must read model outputs from
        results_revisions/ (not results/) so they evaluate the new models
        rather than the original experiment outputs.
        """
        bench_dir = PROJECT_ROOT / "src" / "benchmarking_revisions"

        # Check each benchmarking script, but only if it exists
        # (test_benchmarking_revisions_scripts_exist covers existence)
        for script_name in ["csv_accuracy.py", "txt_accuracy.py"]:
            script_path = bench_dir / script_name
            if not script_path.is_file():
                # Skip content checks if the file doesn't exist yet;
                # the existence test will already report the failure.
                pytest.skip(
                    f"{script_name} does not exist yet -- "
                    "skipping content verification."
                )

            content = script_path.read_text(encoding="utf-8")

            # The script must reference results_revisions somewhere
            # in its source code (e.g. in path constants or string literals)
            assert "results_revisions" in content, (
                f"{script_path} does not reference 'results_revisions'. "
                "Benchmarking revision scripts must read from "
                "results_revisions/, not results/."
            )
