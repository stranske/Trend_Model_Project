"""Regression coverage for script-generated UTC identifiers."""

from __future__ import annotations

import runpy
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest


def test_residual_report_timestamp_uses_utc_with_fixed_clock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    script = Path(__file__).parents[2] / "scripts" / "generate_residual_report.py"
    symbols = runpy.run_path(str(script), run_name="generate_residual_report_test")

    timestamp = symbols["_utc_report_timestamp"](datetime(2026, 8, 13, 17, 5, 9, tzinfo=UTC))

    assert timestamp == "2026-08-13 17:05:09 UTC"


def test_param_sweep_run_id_uses_utc_with_fixed_clock() -> None:
    from scripts.streamlit_param_sweep import _new_run_id

    assert _new_run_id(datetime(2026, 8, 13, 17, 5, 9, tzinfo=UTC)) == "20260813T170509Z"


def test_utc_helpers_normalize_aware_offsets_and_reject_naive_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    script = Path(__file__).parents[2] / "scripts" / "generate_residual_report.py"
    residual_timestamp = runpy.run_path(
        str(script), run_name="generate_residual_report_offset_test"
    )["_utc_report_timestamp"]
    from scripts.streamlit_param_sweep import _new_run_id

    central_time = datetime(2026, 8, 13, 12, 5, 9, tzinfo=timezone(timedelta(hours=-5)))
    assert residual_timestamp(central_time) == "2026-08-13 17:05:09 UTC"
    assert _new_run_id(central_time) == "20260813T170509Z"

    naive = datetime(2026, 8, 13, 17, 5, 9)
    with pytest.raises(ValueError, match="timezone-aware"):
        residual_timestamp(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        _new_run_id(naive)
