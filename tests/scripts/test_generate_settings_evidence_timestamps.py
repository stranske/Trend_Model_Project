"""Regression coverage for settings-evidence timestamps."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from scripts import generate_settings_evidence as evidence


def _assert_aware_utc(timestamp: str) -> None:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == UTC.utcoffset(parsed)


def test_generated_evidence_timestamps_are_aware_utc(monkeypatch) -> None:
    monkeypatch.setattr(evidence, "run_analysis_with_state", lambda *_: {})
    monkeypatch.setattr(evidence, "extract_metric", lambda *_: 1.0)

    result = evidence.run_single_setting_test(
        "lookback_periods",
        {"test_value": 18, "expected_metric": "Sharpe"},
        pd.DataFrame({"asset": [0.01]}),
        {"lookback_periods": 12},
    )

    _assert_aware_utc(result["timestamp"])
    generated_line = next(
        line for line in evidence.generate_summary_report([]).splitlines() if "Generated:" in line
    )
    _assert_aware_utc(generated_line.split("**Generated:** ", 1)[1])


def test_committed_snapshot_provenance_and_scoped_links() -> None:
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    folder = root / "docs/settings_evidence"
    data = json.loads((folder / "provenance.json").read_text())
    for name in ("README.md", "SUMMARY.md"):
        text = (folder / name).read_text()
        assert "Snapshot kind: historical" in text
        assert "2025-12-21" in text
        assert "Source revision: UNKNOWN" in text
    assert data["snapshot_kind"] == "historical"
    assert data["source_revision"] is None
    assert data["source_revision_status"] == "unknown"
    assert data["regenerated"] is False
    assert data["original_aggregate"] == {"PASS": 8, "FAIL": 7, "WARN": 0, "ERROR": 0}
    assert set(data["dispositions"]) == {
        "leverage_cap",
        "max_weight",
        "weighting_scheme",
        "transaction_cost_bps",
        "trend_window",
        "rank_pct",
        "selection_count",
    }
    for name, report in data["reports"].items():
        payload = (folder / name).read_bytes()
        assert payload.startswith(data["report_banner"].encode())
        original = payload.removeprefix(data["report_banner"].encode())
        assert hashlib.sha256(original).hexdigest() == report["sha256_original_body"]
        assert report["source_revision"] is None
    summary = (folder / "SUMMARY.md").read_bytes().removeprefix(data["summary_banner"].encode())
    assert hashlib.sha256(summary).hexdigest() == data["summary_original_body_sha256"]
    readme = (folder / "README.md").read_text()
    for setting, row in data["dispositions"].items():
        assert row["historical_status"] == "FAIL"
        assert row["equivalence"] == "not_established"
        for field in ("entry_point", "fixture_and_mode", "historical_metric", "disposition"):
            assert row[field]
        assert (folder / row["historical_report"]).is_file()
        assert (root / row["current_validation"].split("::")[0]).is_file()
        assert f"[`{setting}`]" in readme
        assert row["disposition"] in readme
