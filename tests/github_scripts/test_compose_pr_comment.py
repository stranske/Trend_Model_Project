from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add script directory to path before importing compose_pr_comment
SCRIPT_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compose_pr_comment  # noqa: E402


def test_compose_pr_comment_with_full_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test comment composition with PR number and SHA."""
    monkeypatch.setenv("SUMMARY_BODY", "All checks passed!")
    monkeypatch.setenv("PR_NUMBER", "123")
    monkeypatch.setenv("HEAD_SHA", "abcdef1234567890")
    monkeypatch.chdir(tmp_path)

    compose_pr_comment.main()

    result = (tmp_path / "gate-summary.md").read_text(encoding="utf-8")
    assert "<!-- gate-summary: pr=123 head=abcdef123456 -->" in result
    assert "All checks passed!" in result

    captured = capsys.readouterr()
    assert "gate-summary: pr=123 head=abcdef123456" in captured.out


def test_compose_pr_comment_with_only_pr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test comment composition with only PR number."""
    monkeypatch.setenv("SUMMARY_BODY", "Status update")
    monkeypatch.setenv("PR_NUMBER", "456")
    monkeypatch.setenv("HEAD_SHA", "")
    monkeypatch.chdir(tmp_path)

    compose_pr_comment.main()

    result = (tmp_path / "gate-summary.md").read_text(encoding="utf-8")
    assert "<!-- gate-summary: pr=456 -->" in result
    assert "Status update" in result


def test_compose_pr_comment_with_fallback_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test comment composition falls back to generic anchor."""
    monkeypatch.setenv("SUMMARY_BODY", "Fallback mode")
    monkeypatch.setenv("PR_NUMBER", "")
    monkeypatch.setenv("HEAD_SHA", "")
    monkeypatch.chdir(tmp_path)

    compose_pr_comment.main()

    result = (tmp_path / "gate-summary.md").read_text(encoding="utf-8")
    assert "<!-- gate-summary: anchor -->" in result
    assert "Fallback mode" in result


def test_compose_pr_comment_requires_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the script exits if summary body is missing."""
    monkeypatch.setenv("SUMMARY_BODY", "")
    monkeypatch.setenv("PR_NUMBER", "789")
    monkeypatch.setenv("HEAD_SHA", "xyz")

    with pytest.raises(SystemExit, match="Summary body missing"):
        compose_pr_comment.main()
