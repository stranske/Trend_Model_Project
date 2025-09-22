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
