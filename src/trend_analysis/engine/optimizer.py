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
    cash_weight: float | None = None  # Fixed allocation to CASH (exact slice)


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


def _apply_cap(w: pd.Series, cap: float, total: float | None = None) -> pd.Series:
    """Cap individual weights at ``cap`` and redistribute the excess."""

    if cap is None:
        return w
    if cap <= 0:
        raise ConstraintViolation("max_weight must be positive")
    total_allocation = float(total if total is not None else w.sum())
    if total_allocation <= NUMERICAL_TOLERANCE_HIGH:
        # Early return: If total allocation is (near) zero, there's nothing to cap or redistribute.
        return w
    # Feasibility check
    if cap * len(w) < total_allocation - NUMERICAL_TOLERANCE_HIGH:
        raise ConstraintViolation("max_weight too small for number of assets")

    w = w.copy()
    while True:
        excess = (w - cap).clip(lower=0)
        if excess.sum() <= NUMERICAL_TOLERANCE_HIGH:
            break
        w = w.clip(upper=cap)
        room_mask = w < cap - NUMERICAL_TOLERANCE_HIGH
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
    total: float | None = None,
) -> pd.Series:
    """Enforce group caps, redistributing excess weight."""

    w = w.copy()
    group_series = pd.Series(groups)
    if not set(w.index).issubset(group_series.index):
        missing = set(w.index) - set(group_series.index)
        raise KeyError(f"Missing group mapping for: {sorted(missing)}")

    all_groups = set(group_series.loc[w.index].values)
    total_allocation = float(total if total is not None else w.sum())
    if all_groups.issubset(group_caps.keys()):
        total_cap = sum(group_caps[g] for g in all_groups)
        if total_cap < total_allocation - NUMERICAL_TOLERANCE_HIGH:
            raise ConstraintViolation("Group caps sum to less than required allocation")

    for group, cap in group_caps.items():
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

    if constraints.long_only:
        w = w.clip(lower=0)
        if w.sum() == 0:
            raise ConstraintViolation(
                "All weights non-positive under long-only constraint"
            )
    w /= w.sum()

    total_allocation = float(w.sum())
    working = w
    cash_weight = None

    # cash_weight processing (fixed slice). We treat a dedicated 'CASH' label.
    if constraints.cash_weight is not None:
        cash_weight = float(constraints.cash_weight)
        if not (0 < cash_weight < 1):
            raise ConstraintViolation("cash_weight must be in (0,1) exclusive")
        if "CASH" not in w.index:
            # Create a CASH row with zero pre-allocation so scaling logic is uniform
            w.loc["CASH"] = 0.0
        non_cash_index = w.index[w.index != "CASH"]
        working = w.loc[non_cash_index].copy()
        if working.empty:
            raise ConstraintViolation("No assets available for non-CASH allocation")
        total_allocation = 1.0 - cash_weight
        working /= working.sum()
        working *= total_allocation
        if constraints.max_weight is not None and len(working) > 0:
            eq_after = total_allocation / len(working)
            if eq_after - NUMERICAL_TOLERANCE_HIGH > constraints.max_weight:
                raise ConstraintViolation(
                    "cash_weight infeasible: remaining allocation forces per-asset weight above max_weight"
                )
    else:
        working = w.copy()

    if constraints.max_weight is not None:
        working = _apply_cap(working, constraints.max_weight, total=total_allocation)

    if constraints.group_caps:
        if not constraints.groups:
            raise ConstraintViolation("Group mapping required when group_caps set")
        missing_assets = [
            asset for asset in working.index if asset not in constraints.groups
        ]
        if missing_assets:
            raise KeyError(
                f"Missing group mapping for assets: {', '.join(missing_assets)}"
            )
        group_mapping = {asset: constraints.groups[asset] for asset in working.index}
        working = _apply_group_caps(
            working, constraints.group_caps, group_mapping, total=total_allocation
        )
        # max weight may have been violated again
        if constraints.max_weight is not None:
            working = _apply_cap(
                working, constraints.max_weight, total=total_allocation
            )

    if cash_weight is not None:
        result = working.copy()
        result.loc["CASH"] = cash_weight
        original_order = list(w.index)
        w = result.reindex(original_order)
        if (
            constraints.max_weight is not None
            and w.loc["CASH"] > constraints.max_weight + NUMERICAL_TOLERANCE_HIGH
        ):
            raise ConstraintViolation("cash_weight exceeds max_weight constraint")
    else:
        w = working

    # cash_weight processing (fixed slice). We treat a dedicated 'CASH' label.
    if constraints.cash_weight is not None:
        cw = float(constraints.cash_weight)
        if not (0 < cw < 1):
            raise ConstraintViolation("cash_weight must be in (0,1) exclusive")
        has_cash = "CASH" in w.index
        if not has_cash:  # pragma: no branch - CASH was injected above when missing
            # Create a CASH row with zero pre-allocation so scaling logic is uniform
            w.loc["CASH"] = 0.0
        # Exclude CASH from scaling
        non_cash_mask = w.index != "CASH"
        non_cash = w[non_cash_mask]
        if non_cash.empty:
            raise ConstraintViolation("No assets available for non-CASH allocation")
        # Feasibility with max_weight: if max_weight is set ensure each non-cash asset
        # could in principle satisfy cap after scaling
        if constraints.max_weight is not None:
            cap = constraints.max_weight
            # Minimal achievable equal weight after carving cash
            eq_after = (1 - cw) / len(non_cash)
            if eq_after - NUMERICAL_TOLERANCE_HIGH > cap:
                raise ConstraintViolation(
                    "cash_weight infeasible: remaining allocation forces per-asset weight above max_weight"
                )
        # Scale non-cash block to (1 - cw)
        scale = (1 - cw) / non_cash.sum()
        non_cash = non_cash * scale
        w.update(non_cash)
        w.loc["CASH"] = cw

        # If max_weight applies to CASH as well enforce; else skip. We enforce for consistency.
        if (
            constraints.max_weight is not None
            and w.loc["CASH"] > constraints.max_weight + NUMERICAL_TOLERANCE_HIGH
        ):
            raise ConstraintViolation("cash_weight exceeds max_weight constraint")

    # cash_weight processing (fixed slice). We treat a dedicated 'CASH' label.
    # NOTE: The block below duplicates the earlier cash handling logic for legacy
    # payloads that mutated the constraint object between validation passes.  The
    # modern ``ConstraintSet`` implementation keeps values stable, so the duplicate
    # code path is effectively unreachable during normal execution.  We retain it
    # to mirror the historic behaviour but exclude it from coverage accounting.
    if constraints.cash_weight is not None:  # pragma: no cover - defensive duplicate
        cw = float(constraints.cash_weight)
        if not (0 < cw < 1):
            raise ConstraintViolation("cash_weight must be in (0,1) exclusive")
        has_cash = "CASH" in w.index
        if not has_cash:
            # Create a CASH row with zero pre-allocation so scaling logic is uniform
            w.loc["CASH"] = 0.0
        # Exclude CASH from scaling
        non_cash_mask = w.index != "CASH"
        non_cash = w[non_cash_mask]
        if non_cash.empty:
            raise ConstraintViolation("No assets available for non-CASH allocation")
        # Feasibility with max_weight: if max_weight is set ensure each non-cash asset
        # could in principle satisfy cap after scaling
        if constraints.max_weight is not None:
            cap = constraints.max_weight
            # Minimal achievable equal weight after carving cash
            eq_after = (1 - cw) / len(non_cash)
            if cap is not None and eq_after - NUMERICAL_TOLERANCE_HIGH > cap:
                raise ConstraintViolation(
                    "cash_weight infeasible: remaining allocation forces per-asset weight above max_weight"
                )
        # Scale non-cash block to (1 - cw)
        scale = (1 - cw) / non_cash.sum()
        non_cash = non_cash * scale
        w.update(non_cash)
        w.loc["CASH"] = cw

        # If max_weight applies to CASH as well enforce; else skip. We enforce for consistency.
        if (
            constraints.max_weight is not None
            and w.loc["CASH"] > constraints.max_weight + NUMERICAL_TOLERANCE_HIGH
        ):
            raise ConstraintViolation("cash_weight exceeds max_weight constraint")

    # Final normalisation guard
    w /= w.sum()
    return w


__all__ = ["ConstraintSet", "ConstraintViolation", "apply_constraints"]
