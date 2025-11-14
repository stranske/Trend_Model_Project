"""Soft coverage for the portfolio constraint optimiser."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.engine.optimizer import (
    ConstraintSet,
    ConstraintViolation,
    _apply_cap,
    _apply_group_caps,
    _redistribute,
    apply_constraints,
)


def test_redistribute_handles_low_mass_uniform_distribution() -> None:
    weights = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
    mask = pd.Series([True, True, True], index=weights.index)
    redistributed = _redistribute(weights.copy(), mask, amount=0.9)
    assert pytest.approx(redistributed.sum(), rel=1e-9) == 0.9
    assert (redistributed > 0).all()


def test_apply_cap_respects_total_and_raises_when_infeasible() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.6})
    capped = _apply_cap(weights, cap=0.5)
    assert capped.max() <= 0.5 + 1e-12

    with pytest.raises(ConstraintViolation):
        _apply_cap(weights, cap=0.2)


def test_apply_group_caps_redistributes_excess() -> None:
    weights = pd.Series({"A": 0.7, "B": 0.3})
    caps = {"tech": 0.6, "other": 0.9}
    groups = {"A": "tech", "B": "other"}
    adjusted = _apply_group_caps(weights, caps, groups)
    assert adjusted.loc["A"] <= 0.6 + 1e-12

    with pytest.raises(KeyError):
        _apply_group_caps(weights, caps, {"A": "tech"})


def test_apply_constraints_supports_mapping_payload_and_cash_weight() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})
    constraints = {
        "long_only": True,
        "max_weight": 0.7,
        "group_caps": {"grp": 1.0},
        "groups": {"A": "grp", "B": "grp"},
        "cash_weight": 0.1,
    }
    result = apply_constraints(weights, constraints)
    assert pytest.approx(result.sum(), rel=1e-9) == 1.0
    assert "CASH" in result.index


def test_apply_constraints_detects_infeasible_long_only() -> None:
    weights = pd.Series({"A": -1.0, "B": -0.5})
    with pytest.raises(ConstraintViolation):
        apply_constraints(weights, ConstraintSet())
