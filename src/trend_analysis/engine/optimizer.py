from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from trend_analysis.constants import NUMERICAL_TOLERANCE_HIGH


class ConstraintViolation(Exception):
    """Raised when a set of constraints is infeasible."""


@dataclass
class ConstraintSet:
    """Configuration for portfolio constraints."""

    long_only: bool = True
    max_weight: float | None = None
    group_caps: Mapping[str, float] | None = None
    groups: Mapping[str, str] | None = None  # asset -> group
    cash_weight: float | None = None


def _redistribute(w: pd.Series, mask: pd.Series, amount: float) -> pd.Series:
    """Redistribute ``amount`` to weights where ``mask`` is True
    proportionally."""

    if amount <= 0:
        return w
    eligible = w[mask]
    if eligible.empty:
        raise ConstraintViolation("No capacity to redistribute excess weight")
    total = float(eligible.sum())
    if total <= NUMERICAL_TOLERANCE_HIGH:
        # If eligible bucket currently has (near) zero mass, distribute uniformly
        share = amount / len(eligible)
        w.loc[eligible.index] += share
    else:
        w.loc[eligible.index] += amount * (eligible / total)
    return w


def _apply_cap(w: pd.Series, cap: float, investable_total: float = 1.0) -> pd.Series:
    """Cap individual weights at ``cap`` and redistribute the excess."""

    if cap is None:
        return w
    cap = float(cap)
    if cap <= 0:
        raise ConstraintViolation("max_weight must be positive")
    if investable_total <= 0:
        raise ConstraintViolation("Target allocation must be positive")

    effective_cap = cap / investable_total
    if effective_cap <= 0:
        raise ConstraintViolation("max_weight must be positive")
    effective_cap = min(effective_cap, 1.0)
    # Feasibility check relative to the required investable capital
    if effective_cap * len(w) < 1 - NUMERICAL_TOLERANCE_HIGH:
        raise ConstraintViolation("max_weight too small for target allocation")

    w = w.copy()
    while True:
        excess = (w - effective_cap).clip(lower=0)
        if excess.sum() <= NUMERICAL_TOLERANCE_HIGH:
            break
        w = w.clip(upper=effective_cap)
        room_mask = w < effective_cap - NUMERICAL_TOLERANCE_HIGH
        # Ensure boolean mask is a Series aligned to w for type safety
        room_mask = (
            pd.Series(room_mask, index=w.index)
            if not isinstance(room_mask, pd.Series)
            else room_mask
        )
        w = _redistribute(w, room_mask, excess.sum())
    return w


def _apply_group_caps(
    w: pd.Series,
    group_caps: Mapping[str, float],
    groups: Mapping[str, str],
    investable_total: float = 1.0,
) -> pd.Series:
    """Enforce group caps, redistributing excess weight."""

    w = w.copy()
    group_series = pd.Series(groups)
    if not set(w.index).issubset(group_series.index):
        missing = set(w.index) - set(group_series.index)
        raise KeyError(f"Missing group mapping for: {sorted(missing)}")

    if investable_total <= NUMERICAL_TOLERANCE_HIGH:
        raise ConstraintViolation("Target allocation must be positive")
    scale = 1.0 / investable_total
    effective_caps: dict[str, float] = {}
    for group, cap in group_caps.items():
        cap = float(cap)
        if cap < 0:
            raise ConstraintViolation(
                f"Group cap for '{group}' must be non-negative"
            )
        effective_caps[group] = min(cap * scale, 1.0)

    all_groups = set(group_series.loc[w.index].values)
    if all_groups.issubset(group_caps.keys()):
        total_cap = sum(effective_caps[g] for g in all_groups)
        if total_cap < 1 - NUMERICAL_TOLERANCE_HIGH:
            raise ConstraintViolation("Group caps sum to less than target allocation")

    for group, cap in effective_caps.items():
        members = group_series[group_series == group].index
        if members.empty:
            continue
        grp_weight = w.loc[members].sum()
        if grp_weight <= cap + NUMERICAL_TOLERANCE_HIGH:
            continue
        excess = grp_weight - cap
        scale = cap / grp_weight
        w.loc[members] *= scale
        others_mask_arr = ~w.index.isin(members)
        others_mask = pd.Series(others_mask_arr, index=w.index)
        w = _redistribute(w, others_mask, excess)
    return w


def apply_constraints(
    weights: pd.Series, constraints: ConstraintSet | Mapping[str, Any]
) -> pd.Series:
    """Project ``weights`` onto the feasible region defined by
    ``constraints``."""

    if isinstance(constraints, Mapping) and not isinstance(constraints, ConstraintSet):
        constraints = ConstraintSet(**constraints)

    w = weights.astype(float).copy()
    if w.empty:
        return w

    investable_total = 1.0
    if constraints.cash_weight is not None:
        cash_weight = float(constraints.cash_weight)
        if cash_weight < 0:
            raise ConstraintViolation("cash_weight must be non-negative")
        if cash_weight >= 1:
            raise ConstraintViolation("cash_weight must be less than 1")
        investable_total = 1.0 - cash_weight
        if investable_total <= NUMERICAL_TOLERANCE_HIGH:
            raise ConstraintViolation("cash_weight leaves no capital for allocation")

    if constraints.long_only:
        w = w.clip(lower=0)
        if w.sum() == 0:
            raise ConstraintViolation(
                "All weights non-positive under long-only constraint"
            )
    w /= w.sum()

    if constraints.max_weight is not None:
        w = _apply_cap(w, constraints.max_weight, investable_total)

    if constraints.group_caps:
        if not constraints.groups:
            raise ConstraintViolation("Group mapping required when group_caps set")
        w = _apply_group_caps(
            w, constraints.group_caps, constraints.groups, investable_total
        )
        # max weight may have been violated again
        if constraints.max_weight is not None:
            w = _apply_cap(w, constraints.max_weight, investable_total)

    # Final normalisation guard
    total = float(w.sum())
    if total <= NUMERICAL_TOLERANCE_HIGH:
        raise ConstraintViolation("Weights sum to zero after applying constraints")
    w *= investable_total / total
    return w


__all__ = ["ConstraintSet", "ConstraintViolation", "apply_constraints"]
