from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HYPOTHETICAL_GOLDEN_CSV = (
    REPO_ROOT / "tests" / "baseline" / "test_golden" / "__hypothetical_fixture__.csv"
)
HYPOTHETICAL_UNRELATED_CSV = (
    REPO_ROOT / "tests" / "baseline" / "__hypothetical_unrelated__.csv"
)


def test_gitignore_keeps_broad_csv_ignore_rule() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    ignore_lines = {line.strip() for line in gitignore.splitlines()}
    assert "*.csv" in ignore_lines


def test_gitignore_has_golden_fixture_exception() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    ignore_lines = {line.strip() for line in gitignore.splitlines()}
    assert "!tests/baseline/test_golden/*.csv" in ignore_lines


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_golden_fixture_pattern_is_not_ignored() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("git metadata not available")

    relative_path = HYPOTHETICAL_GOLDEN_CSV.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "check-ignore", "-v", relative_path],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if result.returncode == 1:
        return
    assert output.startswith(
        ".gitignore:"
    ), f"unexpected check-ignore output: {output or result.stderr.strip()}"
    matching_rule = output.split("\t", 1)[0].split(":", 2)[-1]
    assert matching_rule.startswith("!"), (
        "golden fixture path should not be ignored; " f"matched rule: {matching_rule}"
    )
    assert "test_golden" in matching_rule


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_unrelated_csv_outside_test_golden_is_ignored() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("git metadata not available")

    relative_path = HYPOTHETICAL_UNRELATED_CSV.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v", relative_path],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "unrelated CSV outside test_golden should be ignored; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    output = result.stdout.strip()
    matching_rule = output.split("\t", 1)[0].split(":", 2)[-1]
    assert matching_rule == "*.csv", (
        f"expected broad *.csv ignore rule; matched: {matching_rule}"
    )
