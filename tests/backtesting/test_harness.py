"""Tests for the backtesting harness."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from trend_analysis.backtesting import run_backtest
from trend_analysis.backtesting.harness import (
    BacktestResult,
    _compute_drawdown,
    _compute_metrics,
    _infer_periods_per_year,
    _initial_weights,
    _json_default,
    _normalise_frequency,
    _normalise_weights,
    _prepare_returns,
    _rebalance_calendar,
    _rolling_sharpe,
    _series_to_dict,
    _to_float,
    _weights_to_dict,
)


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


def test_run_backtest_validates_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    base_returns = _synthetic_returns("2020-01-01", 40)
    strategy = MeanWinnerStrategy()

    with pytest.raises(ValueError, match="positive integer"):
        run_backtest(
            base_returns,
            strategy,
            rebalance_freq="M",
            window_size=0,
        )

    with pytest.raises(ValueError, match="window_mode"):
        run_backtest(
            base_returns,
            strategy,
            rebalance_freq="M",
            window_size=5,
            window_mode="diagonal",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="non-negative"):
        run_backtest(
            base_returns,
            strategy,
            rebalance_freq="M",
            window_size=5,
            transaction_cost_bps=-1,
        )

    empty_returns = pd.DataFrame(columns=["A", "B"], index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="numeric columns"):
        run_backtest(
            empty_returns,
            strategy,
            rebalance_freq="M",
            window_size=5,
        )

    with monkeypatch.context() as mp:
        mp.setattr(
            "trend_analysis.backtesting.harness._prepare_returns",
            lambda _: pd.DataFrame(
                columns=["A"], index=pd.DatetimeIndex([], name="Date"), dtype=float
            ),
        )
        with pytest.raises(ValueError, match="at least one row"):
            run_backtest(
                base_returns,
                strategy,
                rebalance_freq="B",
                window_size=2,
            )

    sparse_index = pd.to_datetime(["2020-01-02", "2020-02-02", "2020-03-02"])
    sparse_returns = pd.DataFrame({"A": 0.01, "B": 0.02}, index=sparse_index)
    with monkeypatch.context() as mp:
        mp.setattr(
            "trend_analysis.backtesting.harness._rebalance_calendar",
            lambda *_: pd.DatetimeIndex([]),
        )
        with pytest.raises(ValueError, match="produced no dates"):
            run_backtest(
                sparse_returns,
                strategy,
                rebalance_freq="M",
                window_size=2,
            )

    with pytest.raises(ValueError, match="window_size too large"):
        run_backtest(
            base_returns.iloc[:5],
            strategy,
            rebalance_freq="B",
            window_size=10,
        )


def test_run_backtest_handles_duplicate_index_slices() -> None:
    index = pd.to_datetime(
        [
            "2020-01-01",
            "2020-01-01",
            "2020-01-02",
            "2020-01-02",
            "2020-01-03",
        ]
    )
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.0, 0.03, 0.04], "B": 0.0}, index=index)

    def constant_strategy(window: pd.DataFrame) -> pd.Series:
        return pd.Series(0.5, index=window.columns)

    result = run_backtest(
        returns,
        constant_strategy,
        rebalance_freq="D",
        window_size=2,
        transaction_cost_bps=10,
    )

    # Duplicate index values exercise the slice-handling logic and ensure
    # rebalance windows are still recorded for each date.
    assert result.calendar.tolist() == sorted(result.calendar.tolist())
    assert set(result.training_windows) == set(result.calendar)
    # Verify costs applied once per rebalance despite duplicate slices.
    assert list(result.transaction_costs.index) == list(result.calendar)
    assert np.isclose(result.transaction_costs.iloc[0], 0.001)
    assert np.isclose(result.transaction_costs.iloc[1], 0.0)
    assert np.isclose(result.transaction_costs.iloc[-1], 0.0)


def test_prepare_returns_validates_structure() -> None:
    numeric = _prepare_returns(
        pd.DataFrame(
            {
                "Date": pd.to_datetime(["2020-01-02", "2020-01-01"]),
                "A": [1, 2],
                "Text": ["x", "y"],
            }
        )
    )
    assert list(numeric.columns) == ["A"]
    assert numeric.index.is_monotonic_increasing

    with pytest.raises(ValueError, match="DatetimeIndex"):
        _prepare_returns(pd.DataFrame({"A": [1, 2]}, index=[1, 2]))

    with pytest.raises(ValueError, match="numeric columns"):
        _prepare_returns(
            pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2020-01-01"]),
                    "Text": ["not numeric"],
                }
            )
        )


@pytest.mark.parametrize(
    "freq,expected",
    [
        ("M", "ME"),
        ("ME", "ME"),
        (" q ", "QE"),
        ("2Y", "2YE"),
        ("QM", "QM"),
        ("7", "7"),
    ],
)
def test_normalise_frequency(freq: str, expected: str) -> None:
    assert _normalise_frequency(freq) == expected


def test_rebalance_calendar_intersects_index() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="B")
    calendar = _rebalance_calendar(index, "2B")
    assert calendar.equals(index[1::2])


def test_infer_periods_per_year_handles_various_spacings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily = pd.date_range("2020-01-01", periods=10, freq="B")
    weekly = pd.date_range("2020-01-01", periods=10, freq="W")
    monthly = pd.date_range("2020-01-01", periods=10, freq="M")
    quarterly = pd.date_range("2020-01-01", periods=5, freq="Q")
    sparse = pd.DatetimeIndex([pd.Timestamp("2020-01-01")])
    descending = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-01"),
        ]
    )
    very_sparse = pd.date_range("2020-01-01", periods=2, freq="36M")

    assert _infer_periods_per_year(daily) == 252
    assert _infer_periods_per_year(weekly) == 52
    assert _infer_periods_per_year(monthly) == 12
    assert _infer_periods_per_year(quarterly) == 4
    assert _infer_periods_per_year(sparse) == 1
    assert _infer_periods_per_year(descending) == 1
    assert _infer_periods_per_year(very_sparse) == 1

    with monkeypatch.context() as mp:
        mp.setattr("numpy.diff", lambda _: np.array([], dtype=float))
        assert _infer_periods_per_year(daily) == 1


def test_weight_helpers_cover_all_branches() -> None:
    columns = ["A", "B", "C"]
    initial = _initial_weights(columns, {"A": 0.2, "D": 0.3})
    assert math.isclose(initial.loc["A"], 0.2)
    assert initial.loc["B"] == 0
    assert initial.loc["C"] == 0

    from_series = _normalise_weights(pd.Series({"A": 0.5, "C": 0.5}), columns)
    from_mapping = _normalise_weights({"A": 0.3, "B": 0.7, "D": 1.0}, columns)
    assert math.isclose(from_series.sum(), 1.0)
    assert math.isclose(from_mapping.loc["A"], 0.3)
    assert math.isclose(from_mapping.loc["B"], 0.7)
    assert from_mapping.loc["C"] == 0.0


def test_compute_drawdown_and_metrics_variants() -> None:
    equity = pd.Series([1.0, 1.1, 1.05], index=pd.date_range("2020-01-01", periods=3))
    drawdown = _compute_drawdown(equity)
    assert drawdown.min() <= 0

    returns = pd.Series([0.01, -0.02, 0.03], index=equity.index)
    metrics = _compute_metrics(
        returns,
        equity,
        drawdown,
        periods_per_year=12,
        active_mask=pd.Series([True, True, True], index=equity.index),
    )
    assert set(metrics) == {
        "cagr",
        "volatility",
        "sortino",
        "calmar",
        "max_drawdown",
        "final_value",
        "sharpe",
    }

    empty_metrics = _compute_metrics(
        pd.Series(dtype=float),
        pd.Series([1.0], index=equity.index[:1]),
        pd.Series(dtype=float),
        periods_per_year=0,
        active_mask=pd.Series([False], index=equity.index[:1]),
    )
    assert math.isnan(empty_metrics["cagr"])


def test_rolling_sharpe_and_series_weights_dict_helpers() -> None:
    idx = pd.date_range("2020-01-01", periods=5)
    returns = pd.Series([0.0, 0.01, -0.01, 0.02, -0.02], index=idx)
    sharpe = _rolling_sharpe(returns, periods_per_year=12, window=1)
    assert sharpe.isna().iloc[0]

    assert _series_to_dict(pd.Series(dtype=float)) == {}
    series_dict = _series_to_dict(returns)
    assert series_dict

    weights = pd.DataFrame(
        {
            idx[0]: {"A": 0.0, "B": 0.1},
            idx[1]: {"A": np.nan, "B": 0.0},
        }
    ).T
    weights_dict = _weights_to_dict(weights)
    assert list(weights_dict.keys()) == [idx[0].isoformat()]
    assert _weights_to_dict(pd.DataFrame()) == {}


def test_json_default_and_to_float_helpers() -> None:
    ts = pd.Timestamp("2020-01-01")
    delta = pd.Timedelta(days=1)
    assert _json_default(ts) == ts.isoformat()
    assert _json_default(delta) == delta.isoformat()
    assert math.isclose(_json_default(np.float64(1.2)), 1.2)

    with pytest.raises(TypeError):
        _json_default(object())

    assert math.isnan(_to_float(None))
    assert math.isclose(_to_float(np.float32(1.5)), 1.5)


def test_backtest_result_summary_filters_weights() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    result = BacktestResult(
        returns=pd.Series([0.1, 0.2], index=idx),
        equity_curve=pd.Series([1.0, 1.2], index=idx),
        weights=pd.DataFrame({"A": [0.0, 0.5], "B": [np.nan, 0.0]}, index=idx),
        turnover=pd.Series([0.0, 1.0], index=idx),
        transaction_costs=pd.Series([0.0, 0.001], index=idx),
        rolling_sharpe=pd.Series([np.nan, 1.0], index=idx),
        drawdown=pd.Series([0.0, -0.1], index=idx),
        metrics={"sharpe": 1.0},
        calendar=idx,
        window_mode="rolling",
        window_size=2,
        training_windows={idx[0]: (idx[0], idx[0]), idx[1]: (idx[0], idx[1])},
    )

    summary = result.summary()
    weights_summary = summary["weights"]
    assert list(weights_summary) == [idx[1].isoformat()]
    assert summary["metrics"]["sharpe"] == 1.0
