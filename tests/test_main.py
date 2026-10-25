"""Smoke tests for the Kaggle bootstrap (main.py). CLI must stay dependency-free."""

import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"


def test_main_help_runs_without_dependencies():
    result = subprocess.run(
        [sys.executable, str(MAIN), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--max-samples", "--epochs", "--save-steps", "--no-resume",
                 "--hub-repo", "--max-seq-length"):
        assert flag in result.stdout


def test_main_defaults_pin_t4_sized_run():
    """Help output must reflect the T4 defaults so regressions are visible."""
    result = subprocess.run(
        [sys.executable, str(MAIN), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "20000" in result.stdout      # max-samples row slice
    assert "4096" in result.stdout       # max-seq-length
    assert "200" in result.stdout        # save-steps
