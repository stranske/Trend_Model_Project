from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add script directory to path before importing load_coverage_summary
SCRIPT_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import load_coverage_summary  # noqa: E402


def test_load_coverage_summary_finds_first_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the script finds the first coverage-summary.md file."""
    # Setup
    root = tmp_path / "gate_artifacts" / "downloads"
    summary_dir = root / "python-3.11"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "coverage-summary.md"
    summary_path.write_text("## Coverage: 85%\n", encoding="utf-8")

    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.chdir(tmp_path)

    # Execute
    load_coverage_summary.main()

    # Verify
    result = (tmp_path / "gate-coverage-summary.md").read_text(encoding="utf-8")
    assert result == "## Coverage: 85%\n"

    output = output_file.read_text(encoding="utf-8")
    assert "body<<EOF" in output
    assert "## Coverage: 85%" in output
    assert "EOF" in output


def test_load_coverage_summary_handles_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the script handles missing coverage summary gracefully."""
    root = tmp_path / "gate_artifacts" / "downloads"
    root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    load_coverage_summary.main()

    captured = capsys.readouterr()
    assert "No coverage summary markdown found" in captured.out
    assert not (tmp_path / "gate-coverage-summary.md").exists()


def test_load_coverage_summary_without_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the script works without GITHUB_OUTPUT set."""
    root = tmp_path / "gate_artifacts" / "downloads"
    summary_dir = root / "python-3.12"
    summary_dir.mkdir(parents=True)
    summary_path = summary_dir / "coverage-summary.md"
    summary_path.write_text("Test coverage\n", encoding="utf-8")

    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.chdir(tmp_path)

    load_coverage_summary.main()

    result = (tmp_path / "gate-coverage-summary.md").read_text(encoding="utf-8")
    assert result == "Test coverage\n"
