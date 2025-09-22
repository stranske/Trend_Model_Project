from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.engine.optimizer import (
    ConstraintSet,
    ConstraintViolation,
    _apply_cap,
    _apply_group_caps,
    apply_constraints,
)


def test_apply_cap_returns_early_when_total_is_zero() -> None:
    """Guard against division-by-zero by exercising the early return branch."""

    weights = pd.Series({"A": 0.0, "B": 0.0})

    # ``total`` is explicitly provided as 0 so the function should return
    # immediately without attempting to scale or redistribute.
    result = _apply_cap(weights.copy(), cap=0.5, total=0.0)

    pd.testing.assert_series_equal(result, weights)


def test_apply_group_caps_missing_mapping_raises() -> None:
    """Ensure a clear KeyError is raised when a group is unmapped."""

    weights = pd.Series({"A": 0.6, "B": 0.4})
    group_caps = {"Tech": 0.7}
    groups = {"A": "Tech"}  # Asset ``B`` is intentionally left unmapped.

    with pytest.raises(KeyError) as excinfo:
        _apply_group_caps(weights, group_caps, groups)

    assert "['B']" in str(excinfo.value)


def test_apply_constraints_requires_non_cash_assets_when_cash_weight_set() -> None:
    """Requesting a cash carve-out with no remaining assets should fail."""

    weights = pd.Series({"CASH": 1.0})
    constraints = ConstraintSet(cash_weight=0.3)

    with pytest.raises(ConstraintViolation):
        apply_constraints(weights, constraints)


def test_apply_constraints_rescales_weights_with_cash_and_cap() -> None:
    """Cash allocation should be respected while the remainder obeys max
    weight."""

    weights = pd.Series({"Asset1": 2.0, "Asset2": 1.0, "CASH": 0.0})
    constraints = ConstraintSet(cash_weight=0.25, max_weight=0.6)

    result = apply_constraints(weights, constraints)

    # All weights should sum to 1 and the CASH slice should equal the requested carve-out.
    assert pytest.approx(result.sum(), rel=1e-9) == 1.0
    assert pytest.approx(result.loc["CASH"], rel=1e-9) == 0.25

    # The residual capital should respect the max_weight constraint and maintain order.
    assert (result.drop("CASH") <= 0.6 + 1e-9).all()
    pd.testing.assert_index_equal(result.index, pd.Index(["Asset1", "Asset2", "CASH"]))


@pytest.mark.parametrize("cash_weight", [0.0, 1.0, -0.1, 1.5])
def test_apply_constraints_rejects_invalid_cash_weight(cash_weight: float) -> None:
    """Values outside ``(0, 1)`` should be rejected before any rescaling."""

    weights = pd.Series({"A": 1.0, "B": 2.0})

    with pytest.raises(
        ConstraintViolation, match=r"cash_weight must be in \(0,1\) exclusive"
    ):
        apply_constraints(weights, ConstraintSet(cash_weight=cash_weight))


def test_apply_constraints_reapplies_cap_after_group_redistribution() -> None:
    """Redistribution from group caps should still honour the individual max weight."""

    # Skewed starting weights force substantial redistribution after applying the
    # group cap.  ``max_weight`` must be reapplied to keep the non-capped assets
    # within bounds once they receive the excess allocation.
    weights = pd.Series({"A": 9.0, "B": 0.5, "C": 0.5})
    constraints = ConstraintSet(
        max_weight=0.35,
        group_caps={"G1": 0.15},
        groups={"A": "G1", "B": "G2", "C": "G2"},
    )

    result = apply_constraints(weights, constraints)

    assert pytest.approx(float(result.sum()), rel=1e-9) == 1.0
    assert (result <= 0.35 + 1e-9).all()


def test_apply_constraints_mapping_input_hits_cash_guards() -> None:
    """Mapping inputs exercise the dataclass conversion and cash guard rails."""

    weights = pd.Series({"CASH": 1.0})

    with pytest.raises(
        ConstraintViolation, match="No assets available for non-CASH allocation"
    ):
        apply_constraints(weights, {"cash_weight": 0.2})


def test_apply_constraints_group_caps_and_cash_respect_max_weight() -> None:
    """A cash carve-out combined with group caps should keep all assets under ``max_weight``."""

    weights = pd.Series({"A": 9.0, "B": 0.5, "C": 0.5})
    constraints = {
        "cash_weight": 0.2,
        "max_weight": 0.35,
        "group_caps": {"G1": 0.1},
        "groups": {"A": "G1", "B": "G2", "C": "G2"},
    }

    result = apply_constraints(weights, constraints)

    assert pytest.approx(result.loc["CASH"], rel=1e-9) == 0.2
    assert pytest.approx(result.sum(), rel=1e-9) == 1.0
    assert (result.drop("CASH") <= 0.35 + 1e-9).all()


def test_apply_constraints_cash_weight_infeasible_due_to_cap() -> None:
    """If the remaining allocation breaches ``max_weight`` feasibility, raise ``ConstraintViolation``."""

    weights = pd.Series({"Asset1": 1.0, "Asset2": 1.0})

    with pytest.raises(
        ConstraintViolation,
        match="cash_weight infeasible: remaining allocation forces per-asset weight above max_weight",
    ):
        apply_constraints(weights, ConstraintSet(cash_weight=0.5, max_weight=0.2))


def test_apply_constraints_cash_slice_respects_max_weight_cap() -> None:
    """The dedicated CASH slice must honour ``max_weight`` once reintroduced."""

    weights = pd.Series({"Asset1": 1.0, "Asset2": 1.0})

    with pytest.raises(
        ConstraintViolation,
        match="cash_weight exceeds max_weight constraint",
    ):
        apply_constraints(weights, ConstraintSet(cash_weight=0.6, max_weight=0.5))
