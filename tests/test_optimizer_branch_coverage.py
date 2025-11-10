"""Targeted coverage for ``trend_analysis.engine.optimizer``."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.engine import optimizer


def test_redistribute_ignores_non_positive_amount() -> None:
    weights = pd.Series({"A": 0.2, "B": 0.8})
    mask = weights > 0
    result = optimizer._redistribute(weights.copy(), mask, 0.0)
    pd.testing.assert_series_equal(result, weights)


def test_redistribute_raises_when_no_capacity() -> None:
    weights = pd.Series({"A": 0.2, "B": 0.8})
    with pytest.raises(optimizer.ConstraintViolation):
        optimizer._redistribute(weights.copy(), weights < 0, 0.1)


def test_redistribute_uniform_when_total_zero() -> None:
    weights = pd.Series({"A": 0.0, "B": 0.0})
    mask = pd.Series({"A": True, "B": True})
    result = optimizer._redistribute(weights.copy(), mask, 0.2)
    assert result.sum() == pytest.approx(0.2)
    assert result["A"] == pytest.approx(0.1)
    assert result["B"] == pytest.approx(0.1)


def test_apply_cap_handles_none_and_negative() -> None:
    weights = pd.Series({"A": 0.3, "B": 0.7})
    pd.testing.assert_series_equal(optimizer._apply_cap(weights.copy(), None), weights)
    with pytest.raises(optimizer.ConstraintViolation):
        optimizer._apply_cap(weights.copy(), -0.1)


def test_apply_cap_no_allocation_returns_input() -> None:
    weights = pd.Series({"A": 0.0, "B": 0.0})
    result = optimizer._apply_cap(weights.copy(), 0.5, total=0.0)
    pd.testing.assert_series_equal(result, weights)


def test_apply_group_caps_missing_group_raises() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})
    with pytest.raises(KeyError):
        optimizer._apply_group_caps(weights, {"Tech": 0.7}, {"A": "Tech"})


def test_apply_constraints_accepts_mapping() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.6})
    result = optimizer.apply_constraints(weights, {"long_only": False})
    assert pytest.approx(result.sum()) == 1.0


def test_apply_constraints_empty_weights_returns_empty() -> None:
    empty = pd.Series(dtype=float)
    result = optimizer.apply_constraints(empty, optimizer.ConstraintSet())
    assert result.empty


def test_apply_constraints_long_only_requires_positive_weights() -> None:
    weights = pd.Series({"A": -0.2, "B": -0.1})
    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(weights, optimizer.ConstraintSet(long_only=True))


def test_apply_constraints_cash_weight_validation() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})
    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(weights, optimizer.ConstraintSet(cash_weight=1.2))


def test_apply_constraints_cash_weight_max_cap_infeasible() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5, "CASH": 0.0})
    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(
            weights,
            optimizer.ConstraintSet(cash_weight=0.5, max_weight=0.2),
        )


def test_apply_constraints_group_caps_require_mapping() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(
            weights,
            optimizer.ConstraintSet(group_caps={"Tech": 0.6}),
        )


def test_apply_constraints_group_caps_missing_asset_mapping() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(KeyError):
        optimizer.apply_constraints(
            weights,
            optimizer.ConstraintSet(group_caps={"Tech": 0.6}, groups={"A": "Tech"}),
        )


def test_apply_constraints_cash_weight_without_assets_raises() -> None:
    weights = pd.Series({"CASH": 1.0})
    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(
            weights,
            optimizer.ConstraintSet(cash_weight=0.2, max_weight=0.5),
        )


def test_apply_constraints_cash_weight_exceeds_cap() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})
    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(
            weights,
            optimizer.ConstraintSet(cash_weight=0.6, max_weight=0.5),
        )
