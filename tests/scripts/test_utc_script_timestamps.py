"""Regression coverage for script-generated UTC identifiers."""

from __future__ import annotations

import runpy
from datetime import UTC, datetime
from pathlib import Path


def test_residual_report_timestamp_uses_utc_with_fixed_clock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    script = Path(__file__).parents[2] / "scripts" / "generate_residual_report.py"
    symbols = runpy.run_path(str(script), run_name="generate_residual_report_test")

    timestamp = symbols["_utc_report_timestamp"](datetime(2026, 8, 13, 17, 5, 9, tzinfo=UTC))

    assert timestamp == "2026-08-13 17:05:09 UTC"


def test_param_sweep_run_id_uses_utc_with_fixed_clock() -> None:
    from scripts.streamlit_param_sweep import _new_run_id

    assert _new_run_id(datetime(2026, 8, 13, 17, 5, 9, tzinfo=UTC)) == "20260813T170509Z"
