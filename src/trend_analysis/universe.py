"""Helpers for managing dated universe membership windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


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
        end = (
            pd.to_datetime(raw_end, errors="coerce")
            if raw_end not in (None, "")
            else None
        )
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


def _normalise_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    frame = prices.copy()
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    try:
        date_col = lookup["date"]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError("prices must contain a 'date' column") from exc
    try:
        symbol_col = lookup["symbol"]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError("prices must contain a 'symbol' column") from exc
    rename = {}
    if date_col != "date":
        rename[date_col] = "date"
    if symbol_col != "symbol":
        rename[symbol_col] = "symbol"
    if rename:
        frame = frame.rename(columns=rename)
    if not is_datetime64_any_dtype(frame["date"]):
        frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    return frame


def _normalise_membership_frame(
    membership: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    if isinstance(membership, (str, Path)):
        table = pd.read_csv(membership)
    elif isinstance(membership, pd.DataFrame):
        table = membership.copy()
    else:  # pragma: no cover - defensive guard
        raise TypeError("membership must be a DataFrame or path to a CSV file")
    if table.empty:
        return pd.DataFrame(columns=["symbol", "effective_date", "end_date"])
    lookup = {str(col).strip().lower(): col for col in table.columns}
    name_col = lookup.get("symbol") or lookup.get("fund")
    if name_col is None:
        raise ValueError("membership must include a 'symbol' or 'fund' column")
    eff_col = lookup.get("effective_date")
    if eff_col is None:
        raise ValueError("membership must include an 'effective_date' column")
    end_col = lookup.get("end_date")
    rename = {name_col: "symbol", eff_col: "effective_date"}
    if end_col:
        rename[end_col] = "end_date"
    table = table.rename(columns=rename)
    if "end_date" not in table.columns:
        table["end_date"] = pd.NaT
    table["effective_date"] = pd.to_datetime(table["effective_date"])
    table["end_date"] = pd.to_datetime(table["end_date"])
    if table["effective_date"].isna().any():
        raise ValueError("membership entries must include valid effective dates")
    table["symbol"] = table["symbol"].astype(str)
    return table[["symbol", "effective_date", "end_date"]]


def gate_universe(
    prices: pd.DataFrame,
    membership: pd.DataFrame | str | Path,
    as_of: pd.Timestamp | str,
    *,
    rebalance_only: bool = False,
) -> pd.DataFrame:
    """Filter ``prices`` so only active members remain as-of ``as_of``.

    Parameters
    ----------
    prices:
        Long-form price or return table containing ``date`` and ``symbol`` columns.
    membership:
        Either a DataFrame or a CSV path containing ``symbol``/``fund``,
        ``effective_date`` and optional ``end_date`` columns.
    as_of:
        The rebalance date – rows beyond this timestamp are dropped to avoid
        look-ahead bias.
    rebalance_only:
        When ``True`` the function only returns rows exactly matching ``as_of``.
        Otherwise all rows up to and including ``as_of`` are kept.
    """

    frame = _normalise_price_frame(prices)
    if frame.empty:
        return frame.copy()
    as_of_ts = pd.Timestamp(as_of)
    if pd.isna(as_of_ts):
        raise ValueError("as_of must be a valid timestamp")
    membership_frame = _normalise_membership_frame(membership)
    if membership_frame.empty:
        return frame.iloc[0:0]
    membership_frame = membership_frame[membership_frame["effective_date"] <= as_of_ts]
    if membership_frame.empty:
        return frame.iloc[0:0]
    if rebalance_only:
        window = frame[frame["date"] == as_of_ts]
    else:
        window = frame[frame["date"] <= as_of_ts]
    if window.empty:
        return window.copy()
    merged = window.merge(membership_frame, on="symbol", how="inner")
    if merged.empty:
        return window.iloc[0:0]
    mask = (merged["date"] >= merged["effective_date"]) & (
        merged["end_date"].isna() | (merged["date"] <= merged["end_date"])
    )
    if not mask.any():
        return window.iloc[0:0]
    columns = [col for col in frame.columns if col in merged.columns]
    gated = merged.loc[mask, columns]
    return gated.sort_values(["date", "symbol"]).reset_index(drop=True)


__all__ = [
    "MembershipWindow",
    "MembershipTable",
    "load_universe_membership",
    "apply_membership_windows",
    "gate_universe",
]
