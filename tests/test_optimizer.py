"""Unit tests for :mod:`trend_analysis.engine.optimizer`."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.engine import optimizer


def test_apply_constraints_caps_and_normalises_weights() -> None:
    weights = pd.Series({"A": 0.7, "B": 0.3})

    result = optimizer.apply_constraints(weights, optimizer.ConstraintSet(max_weight=0.6))

    assert pytest.approx(1.0) == float(result.sum())
    assert (result <= 0.6 + optimizer.NUMERICAL_TOLERANCE_HIGH).all()


def test_apply_constraints_long_only_rejects_non_positive_portfolio() -> None:
    weights = pd.Series({"A": -0.5, "B": -0.1})

    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(weights, optimizer.ConstraintSet())


def test_apply_constraints_group_caps_require_group_mapping() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})

    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(
            weights,
            optimizer.ConstraintSet(
                group_caps={"tech": 0.6},
            ),
        )


def test_apply_constraints_group_caps_redistribute_excess_weight() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.2, "C": 0.2})
    constraints = optimizer.ConstraintSet(
        group_caps={"tech": 0.5, "health": 0.6},
        groups={"A": "tech", "B": "tech", "C": "health"},
    )

    result = optimizer.apply_constraints(weights, constraints)

    tech_total = result.loc[["A", "B"]].sum()
    assert tech_total <= 0.5 + optimizer.NUMERICAL_TOLERANCE_HIGH
    assert result.loc["C"] >= 0.2


def test_apply_constraints_cash_weight_adds_cash_and_respects_cap() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})
    constraints = optimizer.ConstraintSet(max_weight=0.45, cash_weight=0.4)

    result = optimizer.apply_constraints(weights, constraints)

    assert pytest.approx(1.0) == float(result.sum())
    assert result.loc["CASH"] == pytest.approx(0.4)
    assert (result.drop("CASH") <= 0.45 + optimizer.NUMERICAL_TOLERANCE_HIGH).all()


def test_apply_constraints_cash_weight_infeasible_with_max_weight() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    constraints = optimizer.ConstraintSet(max_weight=0.2, cash_weight=0.2)

    with pytest.raises(optimizer.ConstraintViolation):
        optimizer.apply_constraints(weights, constraints)


def test_apply_cash_weight_scales_non_cash_and_sets_cash() -> None:
    weights = pd.Series({"A": 0.7, "B": 0.3})

    result = optimizer._apply_cash_weight(weights, cash_weight=0.2, max_weight=None)

    assert result.loc["CASH"] == pytest.approx(0.2)
    assert result.loc["A"] == pytest.approx(0.56)
    assert result.loc["B"] == pytest.approx(0.24)


def test_apply_cash_weight_rejects_missing_non_cash_assets() -> None:
    weights = pd.Series({"CASH": 1.0})

    with pytest.raises(optimizer.ConstraintViolation):
        optimizer._apply_cash_weight(weights, cash_weight=0.2, max_weight=None)


def test_apply_cash_weight_rejects_infeasible_max_weight() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})

    with pytest.raises(optimizer.ConstraintViolation):
        optimizer._apply_cash_weight(weights, cash_weight=0.2, max_weight=0.3)


def test_apply_cap_early_return_when_total_near_zero() -> None:
    weights = pd.Series({"A": 1e-13, "B": 0.0})

    capped = optimizer._apply_cap(weights, cap=0.5)

    pd.testing.assert_series_equal(capped, weights)


def test_redistribute_uniform_when_no_existing_weight() -> None:
    weights = pd.Series({"A": 0.0, "B": 0.0, "C": 1.0})
    mask = pd.Series([True, True, False], index=weights.index)

    redistributed = optimizer._redistribute(weights.copy(), mask, amount=0.2)

    assert redistributed.loc["A"] == pytest.approx(0.1)
    assert redistributed.loc["B"] == pytest.approx(0.1)
    assert redistributed.loc["C"] == pytest.approx(1.0)


def test_redistribute_raises_when_no_capacity_available() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.6})
    mask = pd.Series([False, False], index=weights.index)

    with pytest.raises(optimizer.ConstraintViolation):
        optimizer._redistribute(weights, mask, amount=0.1)
