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
    """Restrict position changes to the provided rebalance dates."""

    if not isinstance(positions.index, pd.DatetimeIndex):
        raise TypeError("positions index must be a DatetimeIndex")

    is_series = isinstance(positions, pd.Series)
    frame = (
        positions.to_frame(name=positions.name or "position")
        if is_series
        else positions.copy()
    )

    schedule = get_rebalance_dates(frame.index, dates)
    if schedule.empty:
        filled = frame.copy()
        filled.loc[:, :] = 0.0
    else:
        mask = frame.index.isin(schedule)
        gated = frame.copy()
        gated.loc[~mask, :] = float("nan")
        filled = gated.ffill().fillna(0.0)

    return filled.iloc[:, 0] if is_series else filled


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


def _match_timezone(idx: pd.DatetimeIndex, template: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if template.tz is None:
        if idx.tz is None:
            return idx
        return idx.tz_convert(None)

    if idx.tz is None:
        return idx.tz_localize(template.tz)
    return idx.tz_convert(template.tz)


__all__ = ["get_rebalance_dates", "apply_rebalance_schedule"]
