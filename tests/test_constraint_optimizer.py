import numpy as np
import pandas as pd
import pytest

from trend_analysis.constants import NUMERICAL_TOLERANCE_HIGH
from trend_analysis.engine.optimizer import ConstraintViolation, apply_constraints


def test_long_only_and_max_weight():
    w = pd.Series([-0.2, 0.6, 0.6], index=["a", "b", "c"])
    out = apply_constraints(w, {"long_only": True, "max_weight": 0.55})
    assert (out >= 0).all()
    assert np.isclose(out.sum(), 1.0)
    assert (out <= 0.55 + NUMERICAL_TOLERANCE_HIGH).all()


def test_group_caps():
    w = pd.Series([0.4, 0.3, 0.3], index=["a", "b", "c"])
    constraints = {
        "long_only": True,
        "max_weight": 0.6,
        "group_caps": {"tech": 0.5},
        "groups": {"a": "tech", "b": "tech", "c": "fin"},
    }
    out = apply_constraints(w, constraints)
    assert np.isclose(out.sum(), 1.0)
    assert out.loc["a"] + out.loc["b"] <= 0.5 + NUMERICAL_TOLERANCE_HIGH


def test_infeasible_max_weight():
    w = pd.Series([0.5, 0.5], index=["a", "b"])
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"max_weight": 0.4})


def test_infeasible_group_caps():
    w = pd.Series([0.5, 0.5], index=["a", "b"])
    constraints = {
        "group_caps": {"g1": 0.4, "g2": 0.4},
        "groups": {"a": "g1", "b": "g2"},
    }
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, constraints)


def test_empty_group_handling():
    """Test that groups with no members are handled correctly."""
    w = pd.Series([0.5, 0.5], index=["a", "b"])
    constraints = {
        "group_caps": {"existing": 1.0, "nonexistent": 0.2},  # nonexistent group
        "groups": {"a": "existing", "b": "existing"},
    }
    out = apply_constraints(w, constraints)
    assert np.isclose(out.sum(), 1.0)
    # Both assets should be in "existing" group, no constraint needed since cap is 1.0
    assert out.loc["a"] + out.loc["b"] <= 1.0 + NUMERICAL_TOLERANCE_HIGH


def test_cash_weight_combined_with_caps():
    w = pd.Series([0.9, 0.1], index=["a", "b"])
    constraints = {
        "max_weight": 0.55,
        "cash_weight": 0.2,
    }
    out = apply_constraints(w, constraints)
    assert np.isclose(out.sum(), 0.8)
    assert out.max() <= 0.55 + NUMERICAL_TOLERANCE_HIGH
