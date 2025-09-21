from __future__ import annotations

import json
import pathlib

from scripts import build_autofix_pr_comment as comment_builder


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
