"""Helpers for managing dated universe membership windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MembershipWindow:
    """Inclusive membership window for a single fund."""

    effective_date: pd.Timestamp
    end_date: pd.Timestamp | None = None

    def active_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """Return a boolean mask marking active rows for ``index``."""

        mask = index >= self.effective_date
        if self.end_date is not None:
            mask &= index <= self.end_date
        return mask


MembershipTable = Mapping[str, Sequence[MembershipWindow]]


def _coerce_timestamp(value: object, *, column: str, fund: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(
            f"Universe membership for '{fund}' is missing a valid {column}."
        )
    return ts


def load_universe_membership(path: str | Path) -> MembershipTable:
    """Load membership windows from ``path`` returning a mapping by fund."""

    membership_csv = Path(path)
    if not membership_csv.exists():
        raise FileNotFoundError(str(membership_csv))

    table = pd.read_csv(membership_csv)
    required = {"fund", "effective_date"}
    missing_cols = required - set(col.strip().lower() for col in table.columns)
    if missing_cols:
        joined = ", ".join(sorted(missing_cols))
        raise ValueError(
            f"Universe membership file '{membership_csv}' is missing columns: {joined}"
        )

    # Normalise column casing for lookups
    rename_map = {col: col.strip().lower() for col in table.columns}
    table = table.rename(columns=rename_map)

    grouped: MutableMapping[str, list[MembershipWindow]] = {}
    for row in table.itertuples(index=False):
        fund = str(getattr(row, "fund", "")).strip()
        if not fund:
            raise ValueError("Universe membership entries must provide a fund name.")
        effective = _coerce_timestamp(
            getattr(row, "effective_date", None), column="effective_date", fund=fund
        )
        raw_end = getattr(row, "end_date", None)
        end = pd.to_datetime(raw_end, errors="coerce") if raw_end not in (None, "") else None
        grouped.setdefault(fund, []).append(MembershipWindow(effective, end))

    ordered: dict[str, tuple[MembershipWindow, ...]] = {}
    for fund, windows in grouped.items():
        ordered[fund] = tuple(sorted(windows, key=lambda item: item.effective_date))
    return ordered


def apply_membership_windows(
    frame: pd.DataFrame, membership: MembershipTable
) -> pd.DataFrame:
    """Return ``frame`` with values outside membership windows nulled."""

    if frame.empty or not membership:
        return frame.copy() if frame.empty else frame

    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            index = pd.to_datetime(frame.index)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("Membership masking requires a DatetimeIndex.") from exc
        masked = frame.copy()
        masked.index = index
    else:
        masked = frame.copy()

    for fund, windows in membership.items():
        if fund not in masked.columns:
            continue
        active = pd.Series(False, index=masked.index)
        for window in windows:
            active |= window.active_mask(masked.index)
        masked.loc[~active, fund] = np.nan
    return masked


__all__ = ["MembershipWindow", "MembershipTable", "load_universe_membership", "apply_membership_windows"]
