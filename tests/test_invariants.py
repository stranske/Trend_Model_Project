"""Tests asserting critical portfolio invariants and transaction cost monotonicity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from trend_analysis.backtesting.harness import run_backtest
from trend_analysis.risk import RiskWindow, compute_constrained_weights


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
    )
    high_cost = run_backtest(
        returns,
        _alternating_strategy,
        rebalance_freq="M",
        window_size=15,
        transaction_cost_bps=120.0,
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
