"""Regression tests for shell script parse linting."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_lint_shell_parse_checks_tracked_scripts() -> None:
    result = subprocess.run(
        ["bash", "scripts/lint_shell.sh"],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert "Shell parse check passed" in result.stdout
