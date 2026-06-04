from __future__ import annotations

import pandas as pd

from trend_analysis.backtesting.harness import _infer_periods_per_year as harness_periods
from trend_analysis.engine.walkforward import _infer_periods_per_year as engine_periods
from trend_analysis.util.frequency import infer_periods_per_year


def test_infer_periods_per_year_reuses_shared_helper() -> None:
    monthly = pd.date_range("2020-01-31", periods=8, freq="ME")

    assert harness_periods is infer_periods_per_year
    assert engine_periods is infer_periods_per_year
    assert harness_periods(monthly) == 12
    assert engine_periods(monthly) == 12


def test_infer_periods_per_year_combines_branch_guards() -> None:
    zero_spacing = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
    sparse = pd.date_range("2020-01-01", periods=2, freq="36ME")
    start = pd.Timestamp("2020-01-01")
    tiny_spacing = pd.DatetimeIndex(
        [
            start,
            start + pd.Timedelta(microseconds=1),
            start + pd.Timedelta(microseconds=2),
        ]
    )

    assert infer_periods_per_year(zero_spacing) == 1
    assert infer_periods_per_year(sparse) == 1
    assert infer_periods_per_year(tiny_spacing) >= 1
