from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend.reporting.unified import generate_unified_report
from trend_analysis import api, export
from trend_analysis.config import load

from tests.test_pipeline_integration_direct import _build_demo_config, _make_returns_frame


def test_multi_period_run_populates_performance_by_regime(tmp_path: Path) -> None:
    returns = _make_returns_frame()
    csv_path = tmp_path / "synthetic.csv"
    returns.to_csv(csv_path, index=False)

    config_data = _build_demo_config(tmp_path, csv_path)
    config_data["regime"] = {
        "enabled": True,
        "proxy": "SPX",
        "lookback": 3,
        "smoothing": 1,
        "min_observations": 2,
    }
    config = load(config_data)

    result = api.run_simulation(config, returns.copy())

    regime_table = result.details["performance_by_regime"]
    assert not regime_table.empty
    assert ("User", "All") in regime_table.columns
    assert ("Equal-Weight", "All") in regime_table.columns
    assert np.isfinite(regime_table.loc["Observations", ("User", "All")])
    assert regime_table.loc["Observations", ("User", "All")] >= 2
    assert result.details["regime_settings"]["min_obs"] == 2

    summary = export.format_summary_text(
        result.details,
        config.sample_split["in_start"],
        config.sample_split["in_end"],
        config.sample_split["out_start"],
        config.sample_split["out_end"],
    )
    assert "Performance by regime" in summary

    report = generate_unified_report(result, config, run_id="multi-period-regime")
    assert "Performance by Regime" in report.html
    assert "Regime analysis unavailable" not in report.html


def test_multi_period_regime_data_normalises_timezone_and_daily_cadence() -> None:
    daily_dates = pd.date_range("2024-01-29", periods=6, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {
            "Date": daily_dates.astype(str),
            "SPX": [0.01, 0.02, -0.01, 0.03, 0.01, -0.02],
        }
    )
    out_index = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])

    prepared = api._prepare_multi_period_regime_data(returns, out_index)

    assert prepared["Date"].tolist() == list(out_index)
    assert prepared["Date"].dt.tz is None
    assert prepared.loc[0, "SPX"] == pytest.approx(1.01 * 1.02 * 0.99 - 1)
    assert prepared.loc[1, "SPX"] == pytest.approx(1.03 * 1.01 * 0.98 - 1)


def test_combine_multi_period_series_prefers_later_period_and_sorts() -> None:
    periods = [
        {"series": pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-02-29", "2024-01-31"]))},
        {"series": pd.Series([3.0, 4.0], index=pd.to_datetime(["2024-02-29", "2024-03-31"]))},
    ]

    combined = api._combine_multi_period_series(periods, "series")

    assert combined is not None
    assert combined.index.tolist() == list(
        pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    )
    assert combined.loc[pd.Timestamp("2024-02-29")] == 3.0
    assert api._combine_multi_period_series([{"other": pd.Series([1.0])}], "series") is None
