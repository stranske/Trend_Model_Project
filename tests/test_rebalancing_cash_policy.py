"""Tests for cash policy handling in rebalancing strategies."""

import numpy as np
import pandas as pd
import pytest

from trend_analysis.rebalancing import (
    CashPolicy,
    DrawdownGuardStrategy,
    DriftBandStrategy,
    PeriodicRebalanceStrategy,
    TurnoverCapStrategy,
    VolTargetRebalanceStrategy,
)


def _cash_policy(explicit_cash: bool, normalize_weights: bool) -> CashPolicy:
    return CashPolicy(
        explicit_cash=explicit_cash,
        cash_return_source="risk_free",
        normalize_weights=normalize_weights,
    )


def _assert_cash_policy(weights: pd.Series, policy: CashPolicy) -> None:
    total = float(weights.sum())
    if policy.explicit_cash:
        assert "CASH" in weights.index
        assert np.isclose(total, 1.0)
    elif policy.normalize_weights:
        assert "CASH" not in weights.index
        assert np.isclose(total, 1.0)
    else:
        assert "CASH" not in weights.index
        assert not np.isclose(total, 1.0)


@pytest.mark.parametrize(
    ("explicit_cash", "normalize_weights"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_turnover_cap_cash_policy(explicit_cash: bool, normalize_weights: bool) -> None:
    strategy = TurnoverCapStrategy({"max_turnover": 0.2, "priority": "largest_gap"})
    current = pd.Series([0.7, 0.3], index=["A", "B"])
    target = pd.Series([0.3, 0.7], index=["A", "B"])
    policy = _cash_policy(explicit_cash, normalize_weights)

    result, _ = strategy.apply(current, target, cash_policy=policy)

    _assert_cash_policy(result, policy)


@pytest.mark.parametrize(
    ("explicit_cash", "normalize_weights"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_drift_band_cash_policy(explicit_cash: bool, normalize_weights: bool) -> None:
    strategy = DriftBandStrategy({"band_pct": 0.2, "min_trade": 0.15, "mode": "partial"})
    current = pd.Series([0.6, 0.35, 0.05], index=["A", "B", "C"])
    target = pd.Series([0.2, 0.35, 0.15], index=["A", "B", "C"])
    policy = _cash_policy(explicit_cash, normalize_weights)

    result, _ = strategy.apply(current, target, cash_policy=policy)

    _assert_cash_policy(result, policy)


@pytest.mark.parametrize(
    ("explicit_cash", "normalize_weights"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_periodic_rebalance_cash_policy(explicit_cash: bool, normalize_weights: bool) -> None:
    strategy = PeriodicRebalanceStrategy({"interval": 1})
    current = pd.Series([0.5, 0.5], index=["A", "B"])
    target = pd.Series([0.3, 0.3], index=["A", "B"])
    policy = _cash_policy(explicit_cash, normalize_weights)

    result, _ = strategy.apply(current, target, cash_policy=policy)

    _assert_cash_policy(result, policy)


@pytest.mark.parametrize(
    ("explicit_cash", "normalize_weights"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_vol_target_cash_policy(explicit_cash: bool, normalize_weights: bool) -> None:
    strategy = VolTargetRebalanceStrategy(
        {"target": 0.2, "window": 2, "lev_min": 0.5, "lev_max": 2.0}
    )
    current = pd.Series([0.6, 0.4], index=["A", "B"])
    target = pd.Series([0.6, 0.4], index=["A", "B"])
    policy = _cash_policy(explicit_cash, normalize_weights)
    equity_curve = [1.0, 1.001, 1.002]

    result, _ = strategy.apply(current, target, cash_policy=policy, equity_curve=equity_curve)

    _assert_cash_policy(result, policy)


@pytest.mark.parametrize(
    ("explicit_cash", "normalize_weights"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_drawdown_guard_cash_policy(explicit_cash: bool, normalize_weights: bool) -> None:
    strategy = DrawdownGuardStrategy({"dd_threshold": 0.1, "guard_multiplier": 0.5, "dd_window": 3})
    current = pd.Series([0.6, 0.4], index=["A", "B"])
    target = pd.Series([0.6, 0.4], index=["A", "B"])
    policy = _cash_policy(explicit_cash, normalize_weights)
    equity_curve = [1.0, 0.9, 0.85]
    rb_state = {}

    result, _ = strategy.apply(
        current,
        target,
        cash_policy=policy,
        equity_curve=equity_curve,
        rb_state=rb_state,
    )

    _assert_cash_policy(result, policy)


def test_vol_target_financing_cost_applies_on_leverage() -> None:
    strategy = VolTargetRebalanceStrategy(
        {
            "target": 0.2,
            "window": 2,
            "lev_min": 0.5,
            "lev_max": 2.0,
            "financing_spread_bps": 50.0,
        }
    )
    current = pd.Series([0.5, 0.5], index=["A", "B"])
    target = pd.Series([0.5, 0.5], index=["A", "B"])
    equity_curve = [1.0, 1.001, 1.002]

    result, cost = strategy.apply(current, target, equity_curve=equity_curve)

    lev = float(result.sum())
    expected = max(0.0, lev - 1.0) * 0.005
    assert cost == pytest.approx(expected)


def test_strategies_sum_to_one_with_explicit_cash() -> None:
    cash_policy = CashPolicy(
        explicit_cash=True, cash_return_source="risk_free", normalize_weights=False
    )
    strategies = [
        (
            TurnoverCapStrategy({"max_turnover": 0.2}),
            pd.Series([0.7, 0.3], index=["A", "B"]),
            pd.Series([0.3, 0.7], index=["A", "B"]),
            {},
        ),
        (
            DriftBandStrategy({"band_pct": 0.2, "min_trade": 0.15, "mode": "partial"}),
            pd.Series([0.6, 0.35, 0.05], index=["A", "B", "C"]),
            pd.Series([0.2, 0.35, 0.15], index=["A", "B", "C"]),
            {},
        ),
        (
            PeriodicRebalanceStrategy({"interval": 1}),
            pd.Series([0.5, 0.5], index=["A", "B"]),
            pd.Series([0.3, 0.3], index=["A", "B"]),
            {},
        ),
        (
            VolTargetRebalanceStrategy({"target": 0.2, "window": 2, "lev_max": 2.0}),
            pd.Series([0.6, 0.4], index=["A", "B"]),
            pd.Series([0.6, 0.4], index=["A", "B"]),
            {"equity_curve": [1.0, 1.001, 1.002]},
        ),
        (
            DrawdownGuardStrategy({"dd_threshold": 0.1, "guard_multiplier": 0.5, "dd_window": 3}),
            pd.Series([0.6, 0.4], index=["A", "B"]),
            pd.Series([0.6, 0.4], index=["A", "B"]),
            {"equity_curve": [1.0, 0.9, 0.85], "rb_state": {}},
        ),
    ]

    for strategy, current, target, kwargs in strategies:
        result, _ = strategy.apply(current, target, cash_policy=cash_policy, **kwargs)
        assert "CASH" in result.index
        assert np.isclose(float(result.sum()), 1.0)
