"""Tests for the backtesting harness."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from trend_analysis.backtesting import run_backtest


class MeanWinnerStrategy:
    """Allocate all weight to the asset with the highest trailing mean
    return."""

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        means = window.mean()
        winner = means.idxmax()
        weights = pd.Series(0.0, index=window.columns)
        weights[winner] = 1.0
        return weights


class AlternatingStrategy:
    """Alternate between two assets to force turnover for testing costs."""

    def __init__(self) -> None:
        self.toggle = False

    def __call__(self, window: pd.DataFrame) -> pd.Series:
        weights = pd.Series(0.0, index=window.columns)
        if self.toggle:
            weights.iloc[1] = 1.0
        else:
            weights.iloc[0] = 1.0
        self.toggle = not self.toggle
        return weights


def _synthetic_returns(start: str, periods: int, freq: str = "B") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq)
    half = periods // 2
    a_returns = np.concatenate(
        [
            np.full(half, 0.001),
            np.full(periods - half, 0.01),
        ]
    )
    b_returns = np.concatenate(
        [
            np.full(half, 0.015),
            np.full(periods - half, -0.004),
        ]
    )
    return pd.DataFrame({"A": a_returns, "B": b_returns}, index=index)


def test_rolling_and_expanding_windows_diverge() -> None:
    returns = _synthetic_returns("2020-01-01", 260)
    strategy = MeanWinnerStrategy()

    expanding = run_backtest(
        returns,
        strategy,
        rebalance_freq="M",
        window_size=60,
        window_mode="expanding",
    )
    rolling = run_backtest(
        returns,
        strategy,
        rebalance_freq="M",
        window_size=60,
        window_mode="rolling",
    )

    assert expanding.equity_curve.iloc[-1] != rolling.equity_curve.iloc[-1]

    for date, (_, window_end) in expanding.training_windows.items():
        assert window_end == date

    summary = expanding.summary()
    expected_metrics = {
        "cagr",
        "volatility",
        "sortino",
        "calmar",
        "max_drawdown",
        "final_value",
        "sharpe",
    }

    assert expanding.metrics.keys() >= expected_metrics

    assert summary["window_mode"] == "expanding"
    assert summary["calendar"]
    first_calendar_entry = summary["calendar"][0]
    assert isinstance(first_calendar_entry, str)
    assert "metrics" in summary
    assert expected_metrics.issubset(summary["metrics"].keys())
    training_windows = summary["training_windows"]
    assert training_windows
    assert first_calendar_entry in training_windows
    first_window = training_windows[first_calendar_entry]
    assert first_window["end"] == first_calendar_entry
    assert first_window["start"] <= first_calendar_entry
    assert "rolling_sharpe" in summary
    assert "turnover" in summary
    assert "transaction_costs" in summary
    assert "returns" in summary
    returns_summary = summary["returns"]
    assert isinstance(returns_summary, dict)
    if returns_summary:
        first_return_value = next(iter(returns_summary.values()))
        assert isinstance(first_return_value, float)
    assert "weights" in summary
    if summary["weights"]:
        first_weight_entry = next(iter(summary["weights"].values()))
        assert isinstance(first_weight_entry, dict)
        if first_weight_entry:
            first_weight_value = next(iter(first_weight_entry.values()))
            assert isinstance(first_weight_value, float)
    assert json.loads(expanding.to_json())["window_mode"] == "expanding"


def test_transaction_costs_applied_to_turnover() -> None:
    index = pd.date_range("2021-01-01", periods=40, freq="B")
    returns = pd.DataFrame({"A": 0.01, "B": -0.005}, index=index)
    strategy = AlternatingStrategy()

    result = run_backtest(
        returns,
        strategy,
        rebalance_freq="W",
        window_size=5,
        transaction_cost_bps=50,
    )

    assert not result.transaction_costs.empty
    first_cost = result.transaction_costs.iloc[0]
    second_cost = result.transaction_costs.iloc[1]
    assert np.isclose(first_cost, 0.005)
    assert np.isclose(second_cost, 0.01)

    active_returns = result.returns.dropna()
    assert np.isclose(active_returns.iloc[0], 0.01 - 0.005)
    assert np.isclose(active_returns.iloc[1], 0.01)

    assert result.turnover.iloc[0] == 1.0
    assert result.turnover.iloc[1] == 2.0
