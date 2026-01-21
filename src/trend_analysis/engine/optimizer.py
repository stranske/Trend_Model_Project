from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
    cash_weight: float | None = None  # Fixed allocation to CASH (exact slice)


def _safe_sum(values: pd.Series | pd.Index | NDArray[np.floating[Any]]) -> float:
    """Sum values without relying on pandas/numpy default sentinels."""

    data = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    return float(np.sum(data, dtype=float, initial=0.0))


def _clip_series(
    w: pd.Series, *, lower: float | None = None, upper: float | None = None
) -> pd.Series:
    """Clip series values using NumPy to avoid pandas mask reductions."""

    values = w.to_numpy(dtype=float)
    if lower is not None:
        values = np.maximum(values, lower)
    if upper is not None:
        values = np.minimum(values, upper)
    return pd.Series(values, index=w.index)


def _redistribute(w: pd.Series, mask: pd.Series, amount: float) -> pd.Series:
    """Redistribute ``amount`` to weights where ``mask`` is True
    proportionally."""

    if amount <= 0:
        return w
    mask_arr = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
    values = w.to_numpy(dtype=float)
    eligible_values = values[mask_arr]
    if eligible_values.size == 0:
        raise ConstraintViolation("No capacity to redistribute excess weight")
    total = _safe_sum(eligible_values)
    if total <= NUMERICAL_TOLERANCE_HIGH:
        # If eligible bucket currently has (near) zero mass, distribute uniformly
        share = amount / len(eligible_values)
        values[mask_arr] += share
    else:
        values[mask_arr] += amount * (eligible_values / total)
    return pd.Series(values, index=w.index)


def _apply_cap(w: pd.Series, cap: float, total: float | None = None) -> pd.Series:
    """Cap individual weights at ``cap`` and redistribute the excess."""

    if cap is None:
        return w
    try:
        cap = float(cap)
    except (TypeError, ValueError) as exc:
        raise ConstraintViolation("max_weight must be numeric") from exc
    if not np.isfinite(cap):
        raise ConstraintViolation("max_weight must be finite")
    if cap <= 0:
        raise ConstraintViolation("max_weight must be positive")
    total_allocation = float(total if total is not None else _safe_sum(w))
    if total_allocation <= NUMERICAL_TOLERANCE_HIGH:
        # Early return: If total allocation is (near) zero, there's nothing to cap or redistribute.
        return w
    # Feasibility check
    if cap * len(w) < total_allocation - NUMERICAL_TOLERANCE_HIGH:
        raise ConstraintViolation("max_weight too small for number of assets")

    w = w.copy()
    while True:
        values = w.to_numpy(dtype=float)
        excess_values = np.maximum(values - cap, 0.0)
        excess_amount = _safe_sum(excess_values)
        if excess_amount <= NUMERICAL_TOLERANCE_HIGH:
            break
        w = pd.Series(np.minimum(values, cap), index=w.index)
        room_mask = w < cap - NUMERICAL_TOLERANCE_HIGH
        # Ensure boolean mask is a Series aligned to w for type safety
        room_mask = (
            pd.Series(room_mask, index=w.index)
            if not isinstance(room_mask, pd.Series)
            else room_mask
        )
        w = _redistribute(w, room_mask, excess_amount)
    return w


def _apply_group_caps(
    w: pd.Series,
    group_caps: Mapping[str, float],
    groups: Mapping[str, str],
    total: float | None = None,
) -> pd.Series:
    """Enforce group caps, redistributing excess weight."""

    w = w.copy()
    if not set(w.index).issubset(groups.keys()):
        missing = set(w.index) - set(groups.keys())
        raise KeyError(f"Missing group mapping for: {sorted(missing)}")

    normalized_caps: dict[str, float] = {}
    for group, cap in group_caps.items():
        try:
            cap_value = float(cap)
        except (TypeError, ValueError) as exc:
            raise ConstraintViolation("group_caps values must be numeric") from exc
        if not np.isfinite(cap_value):
            raise ConstraintViolation("group_caps must be finite")
        if cap_value < 0:
            raise ConstraintViolation("group_caps must be non-negative")
        normalized_caps[group] = cap_value

    group_list = [groups[asset] for asset in w.index]
    all_groups = set(group_list)
    total_allocation = float(total if total is not None else _safe_sum(w))
    if all_groups.issubset(normalized_caps.keys()):
        total_cap = sum(normalized_caps[g] for g in all_groups)
        if total_cap < total_allocation - NUMERICAL_TOLERANCE_HIGH:
            raise ConstraintViolation("Group caps sum to less than required allocation")

    values = w.to_numpy(dtype=float)
    for group, cap in normalized_caps.items():
        members_mask = np.array([grp == group for grp in group_list], dtype=bool)
        if not members_mask.any():
            continue
        grp_weight = _safe_sum(values[members_mask])
        if grp_weight <= cap + NUMERICAL_TOLERANCE_HIGH:
            continue
        excess = grp_weight - cap
        scale = cap / grp_weight
        values[members_mask] *= scale
        w = pd.Series(values, index=w.index)
        others_mask = pd.Series(~members_mask, index=w.index)
        w = _redistribute(w, others_mask, excess)
        values = w.to_numpy(dtype=float)
    return w


def _apply_cash_weight(w: pd.Series, cash_weight: float, max_weight: float | None) -> pd.Series:
    """Scale non-cash weights to accommodate a fixed CASH slice."""

    if not (0 < cash_weight < 1):
        raise ConstraintViolation("cash_weight must be in (0,1) exclusive")

    w = w.copy()
    series_name = w.name
    if "CASH" not in w.index:
        # Create a CASH row with zero pre-allocation so scaling logic is uniform
        w.loc["CASH"] = 0.0

    index = w.index
    non_cash_mask = index != "CASH"
    values = w.to_numpy(dtype=float)
    non_cash_values = values[non_cash_mask]
    if non_cash_values.size == 0:
        raise ConstraintViolation("No assets available for non-CASH allocation")
    non_cash_total = _safe_sum(non_cash_values)
    if non_cash_total <= NUMERICAL_TOLERANCE_HIGH:
        raise ConstraintViolation("No assets available for non-CASH allocation")

    if max_weight is not None:
        eq_after = (1 - cash_weight) / len(non_cash_values)
        if eq_after - NUMERICAL_TOLERANCE_HIGH > max_weight:
            raise ConstraintViolation(
                "cash_weight infeasible: remaining allocation forces per-asset weight above max_weight"
            )

    scale = (1 - cash_weight) / non_cash_total
    values[non_cash_mask] = non_cash_values * scale
    values[index == "CASH"] = cash_weight
    w = pd.Series(values, index=index, name=series_name)

    if max_weight is not None and w.loc["CASH"] > max_weight + NUMERICAL_TOLERANCE_HIGH:
        raise ConstraintViolation("cash_weight exceeds max_weight constraint")

    return w


def apply_constraints(
    weights: pd.Series, constraints: ConstraintSet | Mapping[str, Any]
) -> pd.Series:
    """Project ``weights`` onto the feasible region defined by
    ``constraints``. When ``cash_weight`` is provided, a CASH carve-out is
    applied before caps/group redistribution and revalidated afterward for
    non-dataclass constraint objects that might mutate between passes."""

    revalidate_cash_weight = not isinstance(constraints, (ConstraintSet, Mapping))
    if isinstance(constraints, Mapping) and not isinstance(constraints, ConstraintSet):
        constraints = ConstraintSet(**constraints)

    w = weights.astype(float).copy()
    series_name = w.name
    if w.empty:
        return w
    values = w.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ConstraintViolation("Weights must be finite")

    if constraints.long_only:
        w = _clip_series(w, lower=0)
        total_weight = _safe_sum(w)
        if total_weight == 0:
            raise ConstraintViolation("All weights non-positive under long-only constraint")
    else:
        total_weight = _safe_sum(w)
        if abs(total_weight) <= NUMERICAL_TOLERANCE_HIGH:
            raise ConstraintViolation("Total weight must be non-zero")

    w /= total_weight

    total_allocation = _safe_sum(w)
    working = w
    cash_weight = None
    original_order = list(w.index)

    # cash_weight processing (fixed slice). We treat a dedicated 'CASH' label.
    if constraints.cash_weight is not None:
        cash_weight = float(constraints.cash_weight)
        w = _apply_cash_weight(w, cash_weight, constraints.max_weight)
        total_allocation = 1.0 - cash_weight
        working = w.loc[w.index != "CASH"].copy()
        original_order = list(w.index)
    else:
        working = w.copy()

    if constraints.max_weight is not None:
        working = _apply_cap(working, constraints.max_weight, total=total_allocation)

    if constraints.group_caps:
        if not constraints.groups:
            raise ConstraintViolation("Group mapping required when group_caps set")
        missing_assets = [asset for asset in working.index if asset not in constraints.groups]
        if missing_assets:
            raise KeyError(f"Missing group mapping for assets: {', '.join(missing_assets)}")
        group_mapping = {asset: constraints.groups[asset] for asset in working.index}
        working = _apply_group_caps(
            working, constraints.group_caps, group_mapping, total=total_allocation
        )
        # max weight may have been violated again
        if constraints.max_weight is not None:
            working = _apply_cap(working, constraints.max_weight, total=total_allocation)

    if cash_weight is not None:
        result = working.copy()
        result.loc["CASH"] = cash_weight
        w = result.reindex(original_order)
    else:
        w = working

    if revalidate_cash_weight and constraints.cash_weight is not None:
        cash_weight = float(constraints.cash_weight)
        w = _apply_cash_weight(w, cash_weight, constraints.max_weight)

    # Final normalisation guard
    w /= _safe_sum(w)
    w.name = series_name
    return w


__all__ = ["ConstraintSet", "ConstraintViolation", "apply_constraints"]
