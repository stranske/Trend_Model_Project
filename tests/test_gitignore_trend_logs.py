from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_excludes_trend_nl_logs() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    ignore_lines = {line.strip() for line in gitignore.splitlines()}
    assert "/.trend_nl_logs/" in ignore_lines or ".trend_nl_logs/" in ignore_lines


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_no_tracked_trend_nl_logs() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("git metadata not available")
    result = subprocess.run(
        ["git", "ls-files", "--", ".trend_nl_logs"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""
