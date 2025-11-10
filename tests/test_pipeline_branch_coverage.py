"""Additional branch coverage tests for ``trend_analysis.pipeline`` helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_analysis.core.rank_selection import RiskStatsConfig
from trend_analysis.pipeline import _preprocessing_summary, _run_analysis


def test_preprocessing_summary_marks_month_end() -> None:
    """Monthly cadence text should highlight month-end timestamps."""

    summary = _preprocessing_summary("M", normalised=False, missing_summary=None)

    assert "Cadence: Monthly (month-end)" in summary
    assert ";" not in summary


def test_run_analysis_respects_na_cfg_and_warmup() -> None:
    """Na-as-zero configuration keeps incomplete funds while warmup zeroes rows."""

    dates = pd.date_range("2024-01-31", periods=6, freq="M")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, np.nan, 0.02, 0.015, 0.018, 0.02],
            "FundB": [0.005, 0.006, np.nan, 0.007, np.nan, 0.009],
            "Cash": [0.0003, 0.0004, 0.00035, 0.00037, 0.00036, 0.00034],
        }
    )

    stats_cfg = RiskStatsConfig()
    stats_cfg.metrics_to_run = list(stats_cfg.metrics_to_run) + ["AvgCorr"]
    stats_cfg.na_as_zero_cfg = {  # type: ignore[attr-defined]
        "enabled": True,
        "max_missing_per_window": 2,
        "max_consecutive_gap": 1,
    }

    result = _run_analysis(
        frame,
        "2024-01",
        "2024-03",
        "2024-04",
        "2024-06",
        target_vol=0.1,
        monthly_cost=0.0,
        warmup_periods=1,
        stats_cfg=stats_cfg,
    )

    assert result is not None
    assert result["selected_funds"] == ["FundA", "FundB"]
    assert result["preprocessing_summary"].startswith("Cadence: Monthly (month-end)")
    assert result["in_sample_scaled"].iloc[0].eq(0.0).all()
    assert result["out_sample_scaled"].iloc[0].eq(0.0).all()
    assert all(stat.is_avg_corr is not None for stat in result["in_sample_stats"].values())


def test_run_analysis_returns_none_when_no_value_columns() -> None:
    """Guard clause returns ``None`` when no return columns are available."""

    frame = pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=3, freq="M")})

    result = _run_analysis(
        frame,
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        target_vol=0.1,
        monthly_cost=0.0,
    )

    assert result is None
