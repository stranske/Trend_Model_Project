from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import DateOffset, MonthEnd, Week


def get_rebalance_dates(
    prices_index: pd.Index,
    freq: str | Sequence[object] | pd.DatetimeIndex | Iterable[object],
) -> pd.DatetimeIndex:
    """Return rebalance dates aligned to the provided price index.

    Parameters
    ----------
    prices_index:
        Index of observed trading days. Must be coercible to ``DatetimeIndex``.
    freq:
        Either a pandas frequency string (e.g. ``"M"`` or ``"W"``) / human
        friendly alias (``"monthly"``, ``"weekly"``) or an explicit iterable of
        dates. When an iterable is provided, the resulting rebalance calendar is
        the intersection of those dates with ``prices_index``.
    """

    index = _coerce_datetime_index(prices_index, name="prices")

    if isinstance(freq, str):
        offset = _offset_from_frequency(freq)
        resampled = index.to_series().resample(offset).last().dropna()
        calendar = pd.DatetimeIndex(resampled, name="rebalance_date")
        return calendar.intersection(index)

    if isinstance(freq, pd.DatetimeIndex):
        custom = freq
    elif isinstance(freq, (pd.Series, pd.Index)):
        custom = pd.DatetimeIndex(freq)
    elif isinstance(freq, Iterable):
        custom = pd.DatetimeIndex(pd.to_datetime(list(freq)))
    else:
        raise TypeError(
            "freq must be a string frequency alias or iterable of datelike values"
        )

    custom = _match_timezone(pd.DatetimeIndex(custom), index)
    custom = pd.DatetimeIndex(sorted(set(custom)), name="rebalance_date")
    return custom.intersection(index)


def apply_rebalance_schedule(
    positions: pd.DataFrame | pd.Series,
    dates: Sequence[pd.Timestamp] | pd.DatetimeIndex | Iterable[object],
) -> pd.DataFrame | pd.Series:
    """Restrict position changes to the provided rebalance dates.

    The first observation is always treated as a rebalance so the initial
    portfolio state is preserved even when the provided calendar starts later
    in the window. Raises ``ValueError`` when none of the requested dates align
    with the supplied index so callers can surface configuration errors early.
    """

    if not isinstance(positions.index, pd.DatetimeIndex):
        raise TypeError("positions index must be a DatetimeIndex")

    is_series = isinstance(positions, pd.Series)
    frame = (
        positions.to_frame(name=positions.name or "position")
        if is_series
        else positions.copy()
    )

    original_index = frame.index
    needs_resort = not original_index.is_monotonic_increasing
    if needs_resort:
        frame = frame.sort_index()

    schedule = get_rebalance_dates(frame.index, dates)
    if schedule.empty:
        raise ValueError("No rebalance dates overlap with the positions index")

    if frame.index[0] not in schedule:
        schedule = schedule.insert(0, frame.index[0])
    schedule = pd.DatetimeIndex(schedule.sort_values().unique())

    mask = frame.index.isin(schedule)
    groups = mask.astype("int64").cumsum()
    gated = frame.groupby(groups, group_keys=False).transform("first")

    ordered = gated if not needs_resort else gated.reindex(original_index)
    return ordered.iloc[:, 0] if is_series else ordered


def _offset_from_frequency(freq: str) -> DateOffset:
    freq_clean = freq.strip()
    if not freq_clean:
        raise ValueError("freq must be a non-empty string")

    lower = freq_clean.lower()
    if lower in {"m", "me", "month", "monthly", "month_end", "monthend"}:
        return MonthEnd()
    if lower in {"w", "week", "weekly"}:
        return Week(weekday=4)

    try:
        return to_offset(freq_clean.upper())
    except ValueError as exc:
        raise ValueError(f"Unsupported frequency alias: {freq}") from exc


def _coerce_datetime_index(index: pd.Index, *, name: str) -> pd.DatetimeIndex:
    try:
        dt_index = pd.DatetimeIndex(index)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"{name} must be convertible to a DatetimeIndex") from exc
    if dt_index.empty:
        return dt_index
    return pd.DatetimeIndex(dt_index.sort_values().unique())


def _match_timezone(
    idx: pd.DatetimeIndex, template: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    if template.tz is None:
        if idx.tz is None:
            return idx
        return idx.tz_convert(None)

    if idx.tz is None:
        return idx.tz_localize(template.tz)
    return idx.tz_convert(template.tz)


__all__ = ["get_rebalance_dates", "apply_rebalance_schedule"]
