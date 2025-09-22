import pandas as pd
import pytest

from trend_analysis.engine.optimizer import ConstraintSet, ConstraintViolation, apply_constraints


def test_apply_constraints_rejects_cash_weight_outside_unit_interval() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4}, dtype=float)
    with pytest.raises(ConstraintViolation, match=r"cash_weight must be in \(0,1\) exclusive"):
        apply_constraints(weights, ConstraintSet(cash_weight=1.0))


def test_apply_constraints_requires_non_cash_assets_when_cash_weight_set() -> None:
    weights = pd.Series({"CASH": 1.0}, dtype=float)
    with pytest.raises(ConstraintViolation, match="No assets available for non-CASH allocation"):
        apply_constraints(weights, ConstraintSet(cash_weight=0.2))


def test_apply_constraints_reapplies_max_weight_after_group_caps() -> None:
    weights = pd.Series({"TechA": 0.9, "TechB": 0.05, "Other": 0.05}, dtype=float)
    constraints = ConstraintSet(
        max_weight=0.5,
        group_caps={"Tech": 0.4},
        groups={"TechA": "Tech", "TechB": "Tech", "Other": "Other"},
    )

    adjusted = apply_constraints(weights, constraints)

    assert pytest.approx(float(adjusted.sum()), rel=0, abs=1e-12) == 1.0
    assert (adjusted <= 0.5 + 1e-12).all()
    assert adjusted.loc["Other"] == pytest.approx(0.5)


@pytest.mark.parametrize("cash_weight", [0.0, -0.1, 1.0])
def test_apply_constraints_rejects_non_unit_cash_weight(cash_weight: float) -> None:
    weights = pd.Series({"FundA": 1.0}, dtype=float)

    with pytest.raises(
        ConstraintViolation, match=r"cash_weight must be in \(0,1\) exclusive"
    ):
        apply_constraints(weights, ConstraintSet(cash_weight=cash_weight))


def test_apply_constraints_detects_cash_weight_infeasible_against_max_weight() -> None:
    weights = pd.Series({"FundA": 0.6, "FundB": 0.4}, dtype=float)

    with pytest.raises(
        ConstraintViolation,
        match="cash_weight infeasible: remaining allocation forces per-asset weight above max_weight",
    ):
        apply_constraints(weights, ConstraintSet(max_weight=0.3, cash_weight=0.25))


def test_apply_constraints_rejects_cash_weight_above_cap_after_scaling() -> None:
    weights = pd.Series({"FundA": 0.7, "FundB": 0.3}, dtype=float)

    with pytest.raises(
        ConstraintViolation, match="cash_weight exceeds max_weight constraint"
    ):
        apply_constraints(weights, ConstraintSet(max_weight=0.45, cash_weight=0.5))
