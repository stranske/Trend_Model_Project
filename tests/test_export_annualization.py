"""Regression coverage for aggregate export annualisation."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from trend_analysis.export import combined_summary_result
from trend_analysis.pipeline import _compute_stats


def _period_result(
    returns: list[float],
    *,
    start: str,
    periods_per_year: int,
) -> dict[str, object]:
    index = pd.date_range(start, periods=len(returns), freq="D")
    frame = pd.DataFrame({"Fund": returns}, index=index)
    risk_free = pd.Series(0.0, index=index)
    stats = _compute_stats(frame, risk_free, periods_per_year=periods_per_year)["Fund"]
    return {
        "in_sample_scaled": frame,
        "out_sample_scaled": frame,
        "ew_weights": {"Fund": 1.0},
        "fund_weights": {"Fund": 1.0},
        "risk_free_in_sample": risk_free,
        "risk_free_out_sample": risk_free,
        "periods_per_year": periods_per_year,
        "in_ew_stats": stats,
        "out_ew_stats": stats,
        "in_user_stats": stats,
        "out_user_stats": stats,
        "in_sample_stats": {"Fund": stats},
        "out_sample_stats": {"Fund": stats},
    }


@pytest.mark.parametrize("periods_per_year", [12, 52, 252])
def test_combined_summary_uses_effective_annualization(periods_per_year: int) -> None:
    first = [0.012, -0.006, 0.018, 0.004]
    second = [-0.003, 0.011, 0.007, -0.002]
    combined = combined_summary_result(
        [
            _period_result(first, start="2024-01-01", periods_per_year=periods_per_year),
            _period_result(second, start="2024-02-01", periods_per_year=periods_per_year),
        ]
    )
    returns = pd.Series(first + second)
    expected = _compute_stats(
        pd.DataFrame({"Fund": returns}),
        pd.Series(0.0, index=returns.index),
        periods_per_year=periods_per_year,
    )["Fund"]
    aggregate_stats = (
        combined["out_sample_stats"]["Fund"],
        combined["out_ew_stats"],
        combined["out_user_stats"],
    )
    for actual in aggregate_stats:
        assert actual.cagr == pytest.approx(expected.cagr)
        assert actual.vol == pytest.approx(expected.vol)
        assert actual.sharpe == pytest.approx(expected.sharpe)
        assert actual.sortino == pytest.approx(expected.sortino)
        assert actual.information_ratio == pytest.approx(expected.information_ratio)


def test_compute_stats_requires_explicit_annualization() -> None:
    parameter = inspect.signature(_compute_stats).parameters["periods_per_year"]
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("value", [None, "12", float("nan"), float("inf"), 0, -12])
def test_combined_summary_rejects_missing_or_invalid_annualization(value: object) -> None:
    result = _period_result([0.01, -0.005], start="2024-01-01", periods_per_year=12)
    result["periods_per_year"] = value

    with pytest.raises(ValueError, match="valid periods_per_year"):
        combined_summary_result([result])


def test_combined_summary_rejects_mixed_annualization() -> None:
    with pytest.raises(ValueError, match="one effective periods_per_year"):
        combined_summary_result(
            [
                _period_result([0.01, -0.005], start="2024-01-01", periods_per_year=12),
                _period_result([0.02, -0.01], start="2024-02-01", periods_per_year=252),
            ]
        )
