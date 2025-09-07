import numpy as np
import pandas as pd
import pytest

from trend_analysis.engine.optimizer import (ConstraintViolation,
                                             apply_constraints)


def test_long_only_projection_normalizes():
    w = pd.Series([0.2, -0.1, 0.9], index=["a", "b", "c"], dtype=float)
    out = apply_constraints(w, {"long_only": True})
    assert (out >= 0).all()
    np.testing.assert_allclose(out.sum(), 1.0)


def test_max_weight_cap_redistributes():
    w = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    out = apply_constraints(w, {"long_only": True, "max_weight": 0.55})
    assert out["a"] <= 0.55 + 1e-9
    np.testing.assert_allclose(out.sum(), 1.0)


def test_too_small_max_weight_infeasible():
    w = pd.Series([0.5, 0.5], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"max_weight": 0.3})


def test_group_caps_enforced_and_redistributed():
    w = pd.Series([0.5, 0.5, 0.0], index=["a", "b", "c"], dtype=float)
    groups = {"a": "X", "b": "X", "c": "Y"}
    out = apply_constraints(
        w,
        {
            "long_only": True,
            "group_caps": {"X": 0.6, "Y": 0.6},
            "groups": groups,
        },
    )
    assert out.loc[["a", "b"]].sum() <= 0.6 + 1e-9
    np.testing.assert_allclose(out.sum(), 1.0)


def test_missing_groups_raises():
    w = pd.Series([0.5, 0.5], index=["a", "b"], dtype=float)
    with pytest.raises(KeyError):
        apply_constraints(w, {"group_caps": {"X": 0.6}, "groups": {"a": "X"}})


def test_redistribute_failure_raises():
    w = pd.Series([0.5, 0.5], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"long_only": True, "max_weight": 0.4})
