from __future__ import annotations

import json
import pathlib

import pytest

from scripts import build_autofix_pr_comment as comment_builder

pytestmark = pytest.mark.cosmetic


def test_build_comment_includes_metrics(tmp_path: pathlib.Path) -> None:
    report = {
        "changed": True,
        "classification": {
            "total": 5,
            "new": 2,
            "allowed": 1,
            "timestamp": "2025-09-21T13:07:52Z",
            "by_code": {"E123": 3, "A001": 2},
        },
    }
    history = [{"remaining": 5}, {"remaining": 3}]
    trend = {
        "remaining_latest": 5,
        "new_latest": 2,
        "remaining_spark": "▁▂▃",
        "new_spark": "▂▃▄",
        "codes": {"E123": {"latest": 3, "spark": "▁▂"}},
    }

    report_path = tmp_path / "report.json"
    history_path = tmp_path / "history.json"
    trend_path = tmp_path / "trend.json"
    report_path.write_text(json.dumps(report))
    history_path.write_text(json.dumps(history))
    trend_path.write_text(json.dumps(trend))

    comment = comment_builder.build_comment(
        report_path=report_path,
        history_path=history_path,
        trend_path=trend_path,
        pr_number="42",
    )

    assert comment.startswith(comment_builder.MARKER)
    assert "# Autofix Status" in comment
    assert "| Status |" in comment
    assert "| History points | 2 |" in comment
    assert "Remaining: **5**" in comment
    assert "### Top residual codes" in comment
    assert "## Current per-code counts" in comment
    assert "autofix-report-pr-42" in comment
    assert comment.rstrip().endswith(comment_builder.MARKER)


def test_build_comment_handles_missing_inputs(tmp_path: pathlib.Path) -> None:
    missing_report = tmp_path / "missing.json"

    comment = comment_builder.build_comment(report_path=missing_report, pr_number="")

    assert comment.startswith(comment_builder.MARKER)
    assert "| Remaining issues | 0 |" in comment
    assert "- No additional artifacts" in comment


def test_build_comment_includes_result_block(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "changed": False,
        "classification": {
            "total": 0,
            "new": 0,
            "allowed": 0,
            "timestamp": "2025-09-21T13:07:52Z",
        },
    }
    trend = {
        "remaining_latest": 0,
        "new_latest": 0,
        "remaining_spark": "▁",
        "new_spark": "▁",
    }

    report_path = tmp_path / "report.json"
    trend_path = tmp_path / "trend.json"
    report_path.write_text(json.dumps(report))
    trend_path.write_text(json.dumps(trend))

    monkeypatch.setenv(
        "AUTOFIX_RESULT_BLOCK",
        "Autofix commit: [deadbeef](https://example.test)\nLabels: `autofix:applied`",
    )

    try:
        comment = comment_builder.build_comment(
            report_path=report_path,
            trend_path=trend_path,
            pr_number="7",
        )
    finally:
        monkeypatch.delenv("AUTOFIX_RESULT_BLOCK", raising=False)

    assert "## Autofix result" in comment
    assert "Autofix commit: [deadbeef](https://example.test)" in comment
    assert "Labels: `autofix:applied`" in comment


def test_build_comment_preserves_multiline_result_block(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"changed": True}))

    monkeypatch.setenv(
        "AUTOFIX_RESULT_BLOCK",
        "\n".join(
            (
                "Patch ready: [autofix-patch-pr-5](https://example.test)",
                "Labels: `autofix:patch`, `autofix:debt`",
                "",
                "Apply locally:",
                "1. Download the artifact.",
                "2. Run `git am < autofix.patch`.",
            )
        ),
    )

    try:
        comment = comment_builder.build_comment(
            report_path=report_path,
            pr_number="5",
        )
    finally:
        monkeypatch.delenv("AUTOFIX_RESULT_BLOCK", raising=False)

    result_section = comment.split("## Autofix result", 1)[1]
    assert "Patch ready: [autofix-patch-pr-5](https://example.test)" in result_section
    assert "Labels: `autofix:patch`, `autofix:debt`" in result_section
    assert "Apply locally:" in result_section
    assert "1. Download the artifact." in result_section
    assert "2. Run `git am < autofix.patch`." in result_section


def test_build_comment_reports_clean_result(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"changed": False}))

    monkeypatch.setenv(
        "AUTOFIX_RESULT_BLOCK",
        "\n".join(
            (
                "No changes required.",
                "Labels: `autofix:clean`, `autofix:debt`",
            )
        ),
    )

    try:
        comment = comment_builder.build_comment(
            report_path=report_path,
            pr_number="11",
        )
    finally:
        monkeypatch.delenv("AUTOFIX_RESULT_BLOCK", raising=False)

    result_section = comment.split("## Autofix result", 1)[1]
    assert "No changes required." in result_section
    assert "Labels: `autofix:clean`, `autofix:debt`" in result_section


def test_build_comment_includes_meta_line(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"classification": {}}))

    monkeypatch.setenv("AUTOFIX_TRIGGER_CONCLUSION", "success")
    monkeypatch.setenv("AUTOFIX_TRIGGER_CLASS", "success")
    monkeypatch.setenv("AUTOFIX_TRIGGER_REASON", "labeled")
    monkeypatch.setenv("AUTOFIX_TRIGGER_PR_HEAD", "feature/demo")

    comment = comment_builder.build_comment(
        report_path=report_path,
        pr_number="101",
    )

    meta_line = "<!-- autofix-meta: conclusion=success reason=labeled head=feature/demo -->"
    assert meta_line in comment
    assert comment.splitlines()[1] == meta_line
