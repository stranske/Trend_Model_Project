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


def test_apply_cap_none_returns_original_series():
    w = pd.Series([0.4, 0.6], index=["a", "b"], dtype=float)
    out = optimizer._apply_cap(w, None)
    assert out is w


def test_apply_cap_rejects_non_positive_cap():
    w = pd.Series([0.5, 0.5], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation, match="max_weight must be positive"):
        optimizer._apply_cap(w, 0.0)


def test_apply_cap_uniformly_redistributes_zero_mass_bucket():
    w = pd.Series([1.0, 0.0, 0.0], index=["a", "b", "c"], dtype=float)
    capped = optimizer._apply_cap(w, 0.6)
    expected = pd.Series([0.6, 0.2, 0.2], index=w.index)
    pd.testing.assert_series_equal(capped, expected, check_exact=False, atol=1e-12, rtol=1e-12)


def test_apply_group_caps_handles_missing_cap_entries():
    w = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    groups = {"a": "X", "b": "Y"}
    # No cap provided for group Y; branch should simply skip enforcement
    out = optimizer._apply_group_caps(w, {"X": 0.8}, groups)
    pd.testing.assert_series_equal(out, w)


def test_apply_group_caps_sum_too_small_raises():
    w = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    groups = {"a": "X", "b": "Y"}
    with pytest.raises(
        ConstraintViolation, match="Group caps sum to less than required allocation"
    ):
        optimizer._apply_group_caps(w, {"X": 0.3, "Y": 0.4}, groups)


def test_apply_group_caps_skips_empty_groups():
    w = pd.Series([0.5, 0.5], index=["a", "b"], dtype=float)
    groups = {"a": "X", "b": "Y"}
    caps = {"X": 0.7, "Y": 0.7, "Z": 0.5}
    out = optimizer._apply_group_caps(w, caps, groups)
    np.testing.assert_allclose(out.sum(), 1.0)


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


def test_apply_constraints_accepts_constraintset_instance():
    w = pd.Series([0.8, -0.4, 0.6], index=["a", "b", "c"], dtype=float)
    constraints = optimizer.ConstraintSet(long_only=False)
    out = apply_constraints(w, constraints)
    np.testing.assert_allclose(out.sum(), 1.0)
    assert any(out < 0)


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


def test_cash_weight_added_and_scaled():
    w = pd.Series([0.5, 0.5], index=["a", "b"], dtype=float)
    out = apply_constraints(w, {"cash_weight": 0.1})
    assert "CASH" in out.index
    np.testing.assert_allclose(out.loc["CASH"], 0.1)
    np.testing.assert_allclose(out.loc[["a", "b"]].sum(), 0.9)
    np.testing.assert_allclose(out.sum(), 1.0)


def test_cash_weight_existing_cash_rescaled():
    w = pd.Series([0.4, 0.4, 0.2], index=["a", "b", "CASH"], dtype=float)
    out = apply_constraints(w, {"cash_weight": 0.25})
    np.testing.assert_allclose(out.loc["CASH"], 0.25)
    np.testing.assert_allclose(out.drop("CASH").sum(), 0.75)
    np.testing.assert_allclose(out.sum(), 1.0)


def test_cash_weight_invalid_range_raises():
    w = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"cash_weight": 1.0})
    with pytest.raises(ConstraintViolation):
        apply_constraints(w, {"cash_weight": 0.0})


def test_cash_weight_negative_value_rejected():
    w = pd.Series([0.55, 0.45], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation, match=r"cash_weight must be in \(0,1\) exclusive"):
        apply_constraints(w, {"cash_weight": -0.1})


def test_cash_weight_infeasible_with_max_weight():
    # With two assets, cash=0.3 leaves 0.7. Equal after scaling = 0.35 each which exceeds cap 0.3
    w = pd.Series([0.5, 0.5], index=["a", "b"], dtype=float)
    with pytest.raises(
        ConstraintViolation,
        match="cash_weight infeasible: remaining allocation forces per-asset weight above max_weight",
    ):
        apply_constraints(w, {"cash_weight": 0.3, "max_weight": 0.3})


def test_cash_weight_exceeds_max_weight_for_cash():
    # cash_weight larger than max_weight should raise
    w = pd.Series([0.7, 0.3], index=["a", "b"], dtype=float)
    with pytest.raises(ConstraintViolation, match="exceeds max_weight"):
        apply_constraints(w, {"cash_weight": 0.6, "max_weight": 0.5})


def test_cash_weight_respects_max_weight_after_residual_scaling():
    # Scenario highlighted in review: feasible once residual mass considered
    w = pd.Series([0.6, 0.4], index=["a", "b"], dtype=float)
    out = apply_constraints(w, {"cash_weight": 0.3, "max_weight": 0.4})
    np.testing.assert_allclose(out.loc["CASH"], 0.3)
    assert (out.drop("CASH") <= 0.4 + 1e-9).all()
    np.testing.assert_allclose(out.sum(), 1.0)


def test_cash_weight_requires_non_cash_assets():
    w = pd.Series([1.0], index=["CASH"], dtype=float)
    with pytest.raises(ConstraintViolation, match="No assets available for non-CASH allocation"):
        apply_constraints(w, {"cash_weight": 0.2})
