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
