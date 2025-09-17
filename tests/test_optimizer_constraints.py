import numpy as np
import pandas as pd
import pytest

import trend_analysis.engine.optimizer as optimizer
from trend_analysis.engine.optimizer import ConstraintViolation, apply_constraints


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


def test_redistribute_zero_amount_is_noop():
    w = pd.Series([0.3, 0.7], index=["a", "b"], dtype=float)
    mask = pd.Series([True, False], index=w.index)
    out = optimizer._redistribute(w.copy(), mask, 0.0)
    pd.testing.assert_series_equal(out, w)


def test_redistribute_without_capacity_raises():
    w = pd.Series([0.4, 0.6], index=["a", "b"], dtype=float)
    mask = pd.Series([False, False], index=w.index)
    with pytest.raises(ConstraintViolation):
        optimizer._redistribute(w.copy(), mask, 0.1)


def test_apply_cap_uniformly_redistributes_zero_mass_bucket():
    w = pd.Series([1.0, 0.0, 0.0], index=["a", "b", "c"], dtype=float)
    capped = optimizer._apply_cap(w, 0.6)
    expected = pd.Series([0.6, 0.2, 0.2], index=w.index)
    pd.testing.assert_series_equal(capped, expected, check_exact=False, atol=1e-12, rtol=1e-12)


def test_apply_constraints_empty_input_returns_empty():
    w = pd.Series(dtype=float)
    out = apply_constraints(w, {})
    assert out.empty


def test_apply_constraints_long_only_all_non_positive():
    w = pd.Series([-0.1, -0.2], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"long_only": True})


def test_apply_constraints_group_caps_require_mapping():
    w = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"group_caps": {"G1": 0.7}})


def test_apply_constraints_reapplies_max_after_group_caps():
    w = pd.Series([0.8, 0.2, 0.0], index=["a", "b", "c"], dtype=float)
    constraints = {
        "long_only": True,
        "max_weight": 0.4,
        "group_caps": {"A": 0.5, "B": 0.6},
        "groups": {"a": "A", "b": "A", "c": "B"},
    }
    out = apply_constraints(w, constraints)
    np.testing.assert_allclose(out.sum(), 1.0)
    assert (out <= 0.4 + 1e-9).all()
    # Ensure redistribution moved weight to the previously zero bucket
    assert out["c"] > 0


def test_cash_weight_scales_investable_allocation():
    w = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    out = apply_constraints(w, {"cash_weight": 0.25})
    np.testing.assert_allclose(out.sum(), 0.75)
    np.testing.assert_allclose(out["a"], 0.45)


def test_cash_weight_invalid_range_raises():
    w = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"cash_weight": -0.1})
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"cash_weight": 1.0})


def test_cash_weight_respected_with_other_caps():
    w = pd.Series([0.9, 0.05, 0.05], index=["a", "b", "c"], dtype=float)
    constraints = {
        "max_weight": 0.6,
        "group_caps": {"X": 0.6, "Y": 0.5},
        "groups": {"a": "X", "b": "X", "c": "Y"},
        "cash_weight": 0.2,
    }
    out = apply_constraints(w, constraints)
    np.testing.assert_allclose(out.sum(), 0.8)
    assert out.loc[["a", "b"]].sum() <= 0.6 + 1e-9
    assert out.max() <= 0.6 + 1e-9
