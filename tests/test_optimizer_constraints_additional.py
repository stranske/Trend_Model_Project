from __future__ import annotations

from collections import deque

import pandas as pd
import pytest

import trend_analysis.engine.optimizer as optimizer_mod
from trend_analysis.engine.optimizer import (
    ConstraintSet,
    ConstraintViolation,
    _apply_cap,
    _apply_cash_weight,
    _apply_group_caps,
    _safe_sum,
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
        apply_constraints(weights, constraints)  # type: ignore[arg-type]


def test_apply_constraints_rescales_weights_with_cash_and_cap() -> None:
    """Cash allocation should be respected while the remainder obeys max
    weight."""

    weights = pd.Series({"Asset1": 2.0, "Asset2": 1.0, "CASH": 0.0})
    constraints = ConstraintSet(cash_weight=0.25, max_weight=0.6)

    result = apply_constraints(weights, constraints)

    # All weights should sum to 1 and the CASH slice should equal the requested carve-out.
    assert pytest.approx(_safe_sum(result), rel=1e-9) == 1.0
    assert pytest.approx(result.loc["CASH"], rel=1e-9) == 0.25

    # The residual capital should respect the max_weight constraint and maintain order.
    assert (result.drop("CASH") <= 0.6 + 1e-9).all()
    pd.testing.assert_index_equal(result.index, pd.Index(["Asset1", "Asset2", "CASH"]))


@pytest.mark.parametrize("cash_weight", [0.0, 1.0, -0.1, 1.5])
def test_apply_constraints_rejects_invalid_cash_weight(cash_weight: float) -> None:
    """Values outside ``(0, 1)`` should be rejected before any rescaling."""

    weights = pd.Series({"A": 1.0, "B": 2.0})

    with pytest.raises(ConstraintViolation, match=r"cash_weight must be in \(0,1\) exclusive"):
        apply_constraints(weights, ConstraintSet(cash_weight=cash_weight))


def test_apply_constraints_reapplies_cap_after_group_redistribution() -> None:
    """Redistribution from group caps should still honour the individual max
    weight."""

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

    assert pytest.approx(_safe_sum(result), rel=1e-9) == 1.0
    assert (result <= 0.35 + 1e-9).all()


def test_apply_constraints_mapping_input_hits_cash_guards() -> None:
    """Mapping inputs exercise the dataclass conversion and cash guard
    rails."""

    weights = pd.Series({"CASH": 1.0})

    with pytest.raises(ConstraintViolation, match="No assets available for non-CASH allocation"):
        apply_constraints(weights, {"cash_weight": 0.2})


def test_apply_constraints_group_caps_and_cash_respect_max_weight() -> None:
    """A cash carve-out combined with group caps should keep all assets under
    ``max_weight``."""

    weights = pd.Series({"A": 9.0, "B": 0.5, "C": 0.5})
    constraints = {
        "cash_weight": 0.2,
        "max_weight": 0.35,
        "group_caps": {"G1": 0.1},
        "groups": {"A": "G1", "B": "G2", "C": "G2"},
    }

    result = apply_constraints(weights, constraints)

    assert pytest.approx(result.loc["CASH"], rel=1e-9) == 0.2
    assert pytest.approx(_safe_sum(result), rel=1e-9) == 1.0
    assert (result.drop("CASH") <= 0.35 + 1e-9).all()


def test_apply_constraints_cash_weight_infeasible_due_to_cap() -> None:
    """If the remaining allocation breaches ``max_weight`` feasibility, raise
    ``ConstraintViolation``."""

    weights = pd.Series({"Asset1": 1.0, "Asset2": 1.0})

    with pytest.raises(
        ConstraintViolation,
        match="cash_weight infeasible: remaining allocation forces per-asset weight above max_weight",
    ):
        apply_constraints(weights, ConstraintSet(cash_weight=0.5, max_weight=0.2))


def test_apply_constraints_cash_slice_respects_max_weight_cap() -> None:
    """The dedicated CASH slice must honour ``max_weight`` once
    reintroduced."""

    weights = pd.Series({"Asset1": 1.0, "Asset2": 1.0})

    with pytest.raises(
        ConstraintViolation,
        match="cash_weight exceeds max_weight constraint",
    ):
        apply_constraints(weights, ConstraintSet(cash_weight=0.6, max_weight=0.5))


def test_apply_constraints_enforces_cap_after_group_caps_with_cash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure max-weight caps run both before and after group
    redistribution."""

    weights = pd.Series({"A": 9.0, "B": 0.5, "C": 0.5, "CASH": 0.0})
    constraints = ConstraintSet(
        long_only=True,
        max_weight=0.4,
        group_caps={"G1": 0.15},
        groups={"A": "G1", "B": "G2", "C": "G2"},
        cash_weight=0.2,
    )

    cap_calls = {"count": 0}
    original_apply_cap = optimizer_mod._apply_cap

    def tracking_cap(series: pd.Series, cap: float, total: float | None = None) -> pd.Series:
        cap_calls["count"] += 1
        return original_apply_cap(series, cap, total=total)

    monkeypatch.setattr(optimizer_mod, "_apply_cap", tracking_cap)

    try:
        result = apply_constraints(weights, constraints)
    finally:
        monkeypatch.setattr(optimizer_mod, "_apply_cap", original_apply_cap, raising=False)

    assert cap_calls["count"] >= 2
    assert pytest.approx(result.loc["CASH"], rel=1e-9) == 0.2
    assert pytest.approx(_safe_sum(result), rel=1e-9) == 1.0
    assert (result.drop("CASH") <= 0.4 + 1e-9).all()


def test_apply_cash_weight_scales_non_cash_and_adds_cash() -> None:
    """Helper should scale the non-cash slice to ``1 - cash_weight``."""

    weights = pd.Series({"Asset1": 2.0, "Asset2": 1.0})
    expected_asset1 = 0.8 * 2 / 3
    expected_asset2 = 0.8 * 1 / 3

    adjusted = _apply_cash_weight(weights, cash_weight=0.2, max_weight=0.9)

    assert pytest.approx(adjusted.loc["CASH"], rel=1e-9) == 0.2
    assert pytest.approx(_safe_sum(adjusted), rel=1e-9) == 1.0
    assert pytest.approx(adjusted.loc["Asset1"], rel=1e-9) == expected_asset1
    assert pytest.approx(adjusted.loc["Asset2"], rel=1e-9) == expected_asset2


def test_apply_cash_weight_overwrites_existing_cash_value() -> None:
    """Helper should overwrite any pre-existing CASH weight."""

    weights = pd.Series({"Asset1": 1.0, "CASH": 0.9})

    adjusted = _apply_cash_weight(weights, cash_weight=0.2, max_weight=0.9)

    assert pytest.approx(adjusted.loc["CASH"], rel=1e-9) == 0.2
    assert pytest.approx(adjusted.loc["Asset1"], rel=1e-9) == 0.8
    assert pytest.approx(_safe_sum(adjusted), rel=1e-9) == 1.0


def test_apply_cash_weight_requires_non_cash_assets() -> None:
    """Helper should reject cash-only allocations."""

    weights = pd.Series({"CASH": 1.0})

    with pytest.raises(ConstraintViolation, match="No assets available for non-CASH allocation"):
        _apply_cash_weight(weights, cash_weight=0.2, max_weight=None)


def test_apply_cash_weight_rejects_invalid_cash_weight() -> None:
    """Helper should guard against invalid cash ranges."""

    weights = pd.Series({"A": 1.0})

    with pytest.raises(ConstraintViolation, match=r"cash_weight must be in \(0,1\) exclusive"):
        _apply_cash_weight(weights, cash_weight=1.0, max_weight=None)


def test_apply_cash_weight_infeasible_due_to_max_weight() -> None:
    """Helper should reject infeasible max-weight/cash combinations."""

    weights = pd.Series({"A": 1.0, "B": 1.0})

    with pytest.raises(
        ConstraintViolation,
        match="cash_weight infeasible: remaining allocation forces per-asset weight above max_weight",
    ):
        _apply_cash_weight(weights, cash_weight=0.5, max_weight=0.2)


def test_apply_cash_weight_rejects_cash_above_cap() -> None:
    """Helper should reject cash that breaches the max-weight cap."""

    weights = pd.Series({"A": 1.0, "B": 1.0})

    with pytest.raises(ConstraintViolation, match="cash_weight exceeds max_weight constraint"):
        _apply_cash_weight(weights, cash_weight=0.6, max_weight=0.5)


def test_apply_cash_weight_allows_none_max_weight() -> None:
    """Helper should allow valid cash weights when no cap is provided."""

    weights = pd.Series({"A": 1.0, "B": 1.0})

    adjusted = _apply_cash_weight(weights, cash_weight=0.2, max_weight=None)

    assert pytest.approx(_safe_sum(adjusted), rel=1e-9) == 1.0
    assert pytest.approx(adjusted.loc["CASH"], rel=1e-9) == 0.2


def test_apply_constraints_cash_weight_two_passes_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both cash passes should be idempotent for stable constraints."""

    weights = pd.Series({"A": 0.7, "B": 0.3})
    constraints = ConstraintSet(cash_weight=0.2, max_weight=0.6)

    helper_calls: list[pd.Series] = []
    original_helper = optimizer_mod._apply_cash_weight

    def tracking_helper(
        series: pd.Series, cash_weight: float, max_weight: float | None
    ) -> pd.Series:
        result = original_helper(series, cash_weight, max_weight)
        helper_calls.append(result.copy())
        return result

    monkeypatch.setattr(optimizer_mod, "_apply_cash_weight", tracking_helper)

    try:
        out = apply_constraints(weights, constraints)
    finally:
        monkeypatch.setattr(optimizer_mod, "_apply_cash_weight", original_helper, raising=False)

    assert len(helper_calls) == 2
    pd.testing.assert_series_equal(helper_calls[0], helper_calls[1])
    pd.testing.assert_series_equal(out, helper_calls[-1])


class _DynamicConstraintSet:
    """Constraint-like shim that changes the reported cash weight per
    access."""

    def __init__(self, cash_sequence: list[float | None], **kwargs: object) -> None:
        self._cash_values: deque[float | None] = deque(cash_sequence)
        self.history: list[float | None] = []
        self.long_only = kwargs.get("long_only", True)
        self.max_weight = kwargs.get("max_weight")
        self.group_caps = kwargs.get("group_caps")
        self.groups = kwargs.get("groups")

    @property
    def cash_weight(self) -> float | None:
        value = self._cash_values[0]
        if len(self._cash_values) > 1:
            value = self._cash_values.popleft()
        self.history.append(value)
        return value


def test_cash_weight_revalidation_rejects_out_of_range_values() -> None:
    """The second validation pass should still enforce the allowed range."""

    weights = pd.Series({"A": 0.6, "B": 0.4})
    constraints = _DynamicConstraintSet([0.25, 1.2])

    with pytest.raises(ConstraintViolation, match=r"cash_weight must be in \(0,1\) exclusive"):
        apply_constraints(weights, constraints)  # type: ignore[arg-type]

    assert constraints.history == [0.25, 1.2]


def test_cash_weight_revalidation_detects_infeasible_caps() -> None:
    """Updating the cash slice should re-trigger feasibility checks."""

    weights = pd.Series({"A": 0.6, "B": 0.4})
    constraints = _DynamicConstraintSet([0.5, 0.1], max_weight=0.3)

    with pytest.raises(
        ConstraintViolation,
        match="cash_weight infeasible: remaining allocation forces per-asset weight above max_weight",
    ):
        apply_constraints(weights, constraints)  # type: ignore[arg-type]

    assert constraints.history == [0.5, 0.1]


def test_cash_weight_revalidation_checks_cash_cap() -> None:
    """The final cash assignment must respect the individual max weight."""

    weights = pd.Series({"A": 0.7, "B": 0.3})
    constraints = _DynamicConstraintSet([0.2, 0.6], max_weight=0.5)

    with pytest.raises(ConstraintViolation, match="cash_weight exceeds max_weight"):
        apply_constraints(weights, constraints)  # type: ignore[arg-type]

    assert constraints.history == [0.2, 0.6]
