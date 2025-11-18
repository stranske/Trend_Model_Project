"""Tests asserting critical portfolio invariants and transaction cost monotonicity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from trend.validation import build_validation_frame, validate_prices_frame
from trend_analysis.backtesting.harness import _rebalance_calendar, run_backtest
from trend_analysis.risk import RiskWindow, compute_constrained_weights
from trend_analysis.universe import gate_universe
from trend_analysis.util.rolling import rolling_shifted


def _business_index(start: str = "2024-01-02", periods: int = 90) -> pd.DatetimeIndex:
    calendar = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.date_range(start=start, periods=periods, freq=calendar)


def _risk_returns() -> pd.DataFrame:
    idx = _business_index(periods=40)
    data = {
        "Alpha": np.linspace(0.01, 0.03, len(idx)),
        "Beta": np.linspace(-0.005, 0.015, len(idx)),
        "Gamma": np.linspace(0.002, 0.022, len(idx)),
    }
    return pd.DataFrame(data, index=idx)


def _backtest_returns() -> pd.DataFrame:
    idx = _business_index(periods=80)
    alpha = np.where(np.arange(len(idx)) % 2 == 0, 0.012, -0.004)
    beta = np.where(np.arange(len(idx)) % 2 == 0, -0.003, 0.011)
    return pd.DataFrame({"Alpha": alpha, "Beta": beta}, index=idx)


def _alternating_strategy(window: pd.DataFrame) -> pd.Series:
    last = window.iloc[-1]
    if float(last["Alpha"]) >= float(last["Beta"]):
        return pd.Series({"Alpha": 0.75, "Beta": 0.25})
    return pd.Series({"Alpha": 0.25, "Beta": 0.75})


def test_compute_constrained_weights_produces_finite_normalised_weights() -> None:
    returns = _risk_returns().iloc[:24]
    base_weights = {"Alpha": 0.5, "Beta": 0.3, "Gamma": 0.2}
    previous_weights = {"Alpha": 0.2, "Beta": 0.5, "Gamma": 0.3}
    max_turnover = 0.35

    weights, diagnostics = compute_constrained_weights(
        base_weights,
        returns,
        window=RiskWindow(length=6, decay="simple"),
        target_vol=0.15,
        periods_per_year=12.0,
        floor_vol=0.02,
        long_only=True,
        max_weight=0.7,
        previous_weights=previous_weights,
        max_turnover=max_turnover,
    )

    assert np.isfinite(weights.to_numpy()).all()
    assert np.isclose(float(weights.sum()), 1.0)
    assert diagnostics.turnover_value <= max_turnover + 1e-12
    assert diagnostics.turnover.index.name == "rebalance"


def test_backtest_calendar_respects_business_days() -> None:
    returns = _backtest_returns()
    result = run_backtest(
        returns,
        _alternating_strategy,
        rebalance_freq="M",
        window_size=15,
        transaction_cost_bps=0.0,
        min_trade=0.0,
    )

    holidays = set(
        USFederalHolidayCalendar().holidays(
            start=returns.index.min(), end=returns.index.max()
        )
    )

    for ts in result.calendar:
        assert ts in returns.index
        assert ts.weekday() < 5
        assert ts not in holidays


def test_transaction_costs_are_monotonic() -> None:
    returns = _backtest_returns()
    low_cost = run_backtest(
        returns,
        _alternating_strategy,
        rebalance_freq="M",
        window_size=15,
        transaction_cost_bps=0.0,
        min_trade=0.0,
    )
    high_cost = run_backtest(
        returns,
        _alternating_strategy,
        rebalance_freq="M",
        window_size=15,
        transaction_cost_bps=120.0,
        min_trade=0.0,
    )

    combined = pd.concat(
        [
            low_cost.returns.rename("low"),
            high_cost.returns.rename("high"),
        ],
        axis=1,
    ).dropna(how="any")

    assert not high_cost.transaction_costs.empty
    assert (combined["high"] <= combined["low"] + 1e-12).all()
    assert (combined["high"] < combined["low"] - 1e-9).any()


def test_validation_requires_date_and_value_columns() -> None:
    frame = pd.DataFrame({"Alpha": [0.5]})
    with pytest.raises(ValueError, match="must include 'Date'"):
        build_validation_frame(frame)

    date_only = pd.DataFrame({"Date": [pd.Timestamp("2024-01-31")]})
    with pytest.raises(ValueError, match="value columns to validate"):
        build_validation_frame(date_only)

    tidy = build_validation_frame(
        pd.DataFrame({"Date": ["2024-01-31"], "Alpha": [1.0], "Beta": [2.0]})
    )
    with pytest.raises(KeyError, match="symbol"):
        validate_prices_frame(tidy.drop(columns=["symbol"]))


def test_gate_universe_excludes_non_members_at_rebalance() -> None:
    prices = pd.DataFrame(
        {
            "date": [
                "2020-01-10",
                "2020-01-10",
                "2020-01-10",
                "2020-01-20",
                "2020-01-20",
                "2020-01-20",
            ],
            "symbol": [
                "Alpha",
                "Beta",
                "Gamma",
                "Alpha",
                "Beta",
                "Gamma",
            ],
            "close": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
        }
    )
    membership = pd.DataFrame(
        {
            "symbol": ["Alpha", "Beta"],
            "effective_date": ["2020-01-01", "2020-01-15"],
            "end_date": ["2020-01-31", None],
        }
    )

    jan10 = gate_universe(prices, membership, as_of="2020-01-10", rebalance_only=True)
    assert set(jan10["symbol"]) == {"Alpha"}

    jan20 = gate_universe(prices, membership, as_of="2020-01-20", rebalance_only=True)
    assert set(jan20["symbol"]) == {"Alpha", "Beta"}
    assert "Gamma" not in jan20["symbol"].values


def test_rolling_shifted_never_uses_future_data() -> None:
    index = pd.date_range("2024-01-02", periods=8, freq="B")
    series = pd.Series(np.arange(1, len(index) + 1, dtype=float), index=index)

    window = 3
    result = rolling_shifted(series, window=window, agg="mean")
    expected = series.shift(1).rolling(window=window, min_periods=window).mean()

    pdt.assert_series_equal(result, expected)

    for pos in range(window, len(series)):
        current = index[pos]
        history = series.iloc[pos - window : pos]
        assert np.isclose(result.loc[current], history.mean(), equal_nan=True)


def test_transaction_costs_reduce_returns_by_expected_amount() -> None:
    index = pd.date_range("2021-01-01", periods=30, freq="B")
    returns = pd.DataFrame({"Alpha": 0.01, "Beta": 0.005}, index=index)

    class FlipFlopStrategy:
        def __init__(self) -> None:
            self.toggle = False

        def __call__(self, window: pd.DataFrame) -> pd.Series:
            weights = pd.Series(0.0, index=window.columns)
            if self.toggle:
                weights.iloc[0] = 1.0
            else:
                weights.iloc[1] = 1.0
            self.toggle = not self.toggle
            return weights

    baseline = run_backtest(
        returns,
        FlipFlopStrategy(),
        rebalance_freq="W",
        window_size=5,
        transaction_cost_bps=0.0,
        min_trade=0.0,
    )
    costly = run_backtest(
        returns,
        FlipFlopStrategy(),
        rebalance_freq="W",
        window_size=5,
        transaction_cost_bps=100.0,
        min_trade=0.0,
    )

    expected = baseline.returns.copy()
    tx_per_unit = 100.0 / 10000.0
    for ts, turnover in costly.per_period_turnover[
        costly.per_period_turnover > 0
    ].items():
        expected.loc[ts] -= turnover * tx_per_unit

    observed = costly.returns.dropna()
    aligned_expected = expected.loc[observed.index]
    pdt.assert_series_equal(observed, aligned_expected, check_names=False)


def test_backtest_calendar_matches_calendar_helper() -> None:
    returns = _backtest_returns().iloc[:60]
    freq = "M"
    window_size = 12
    result = run_backtest(
        returns,
        _alternating_strategy,
        rebalance_freq=freq,
        window_size=window_size,
        transaction_cost_bps=0.0,
        min_trade=0.0,
    )

    raw_calendar = _rebalance_calendar(returns.index, freq)
    expected_calendar = pd.DatetimeIndex(
        [date for date in raw_calendar if len(returns.loc[:date]) >= window_size]
    )
    pdt.assert_index_equal(result.calendar, expected_calendar)
