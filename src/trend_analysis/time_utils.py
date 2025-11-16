from __future__ import annotations

import calendar
import datetime as dt
from typing import Iterable

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

DEFAULT_TIMEZONE = "UTC"
DEFAULT_FREQUENCY = "B"
DEFAULT_CALENDAR = "simple"


def _observed(day: dt.date) -> dt.date:
    if day.weekday() == 5:  # Saturday -> Friday
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> Monday
        return day + dt.timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    last_day = calendar.monthrange(year, month)[1]
    last = dt.date(year, month, last_day)
    offset = (last.weekday() - weekday) % 7
    return last - dt.timedelta(days=offset)


def _simple_us_holidays(year: int) -> set[dt.date]:
    holidays: set[dt.date] = set()
    holidays.add(_observed(dt.date(year, 1, 1)))  # New Year's Day
    holidays.add(_observed(dt.date(year, 7, 4)))  # Independence Day
    holidays.add(_observed(dt.date(year, 12, 25)))  # Christmas
    holidays.add(_nth_weekday(year, 1, calendar.MONDAY, 3))  # MLK Day
    holidays.add(_nth_weekday(year, 2, calendar.MONDAY, 3))  # Presidents' Day
    holidays.add(_last_weekday(year, 5, calendar.MONDAY))  # Memorial Day
    holidays.add(_nth_weekday(year, 9, calendar.MONDAY, 1))  # Labor Day
    holidays.add(_nth_weekday(year, 11, calendar.THURSDAY, 4))  # Thanksgiving
    return holidays


def _resolve_frequency_tag(
    freq: str | None,
    observed: pd.Series | None = None,
) -> str:
    candidate = freq
    if candidate in (None, "") and observed is not None:
        try:
            inferred = pd.infer_freq(pd.DatetimeIndex(observed).sort_values())
        except Exception:  # pragma: no cover - inference best-effort
            inferred = None
        candidate = inferred
    if not candidate:
        return DEFAULT_FREQUENCY
    tag = str(candidate).upper()
    if tag in {"D", "DAILY"}:
        return "B"
    if tag.startswith("B"):
        return "B"
    if tag in {"W", "WEEKLY"}:
        return "W-FRI"
    if tag.startswith("W-"):
        return tag
    if tag.startswith("M"):
        return "M"
    return tag


def _resolve_holidays(
    start: dt.date,
    end: dt.date,
    calendar_name: str | None,
    overrides: Iterable[pd.Timestamp] | None,
) -> pd.DatetimeIndex:
    if overrides is not None:
        holidays = pd.to_datetime(list(overrides))
        return pd.DatetimeIndex(sorted(h for h in holidays if start <= h.date() <= end))

    cal = (calendar_name or DEFAULT_CALENDAR).lower()
    if cal not in {"simple", "nyse", "us"}:
        cal = DEFAULT_CALENDAR
    year_start = start.year - 1
    year_end = end.year + 1
    collected: list[pd.Timestamp] = []
    for year in range(year_start, year_end + 1):
        for day in _simple_us_holidays(year):
            collected.append(pd.Timestamp(day))
    idx = pd.DatetimeIndex(collected)
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return idx[mask]


def align_calendar(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    frequency: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    holiday_calendar: str | None = DEFAULT_CALENDAR,
    holidays: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    if date_col not in df.columns:
        raise KeyError(f"DataFrame must contain a '{date_col}' column")
    if df.empty:
        result = df.copy()
        result.attrs.setdefault(
            "calendar_alignment",
            {
                "target_frequency": _resolve_frequency_tag(frequency),
                "timezone": DEFAULT_TIMEZONE,
                "calendar": (holiday_calendar or DEFAULT_CALENDAR).lower(),
                "timestamp_count": 0,
            },
        )
        return result

    work = df.copy()
    series = pd.to_datetime(work[date_col], errors="coerce")
    mask = series.notna()
    if not mask.any():
        raise ValueError(f"Column '{date_col}' contains no valid timestamps")
    work = work.loc[mask].copy()
    series = pd.DatetimeIndex(series[mask])

    tz = timezone or DEFAULT_TIMEZONE
    if series.tz is None:
        series = series.tz_localize(tz)
    else:
        series = series.tz_convert(tz)
    series = series.tz_convert(DEFAULT_TIMEZONE).tz_localize(None)
    work[date_col] = series

    work.sort_values(date_col, inplace=True)
    work.drop_duplicates(subset=[date_col], inplace=True)

    freq_tag = _resolve_frequency_tag(frequency, work[date_col])
    weekend_dropped = 0
    holiday_dropped = 0

    if freq_tag == "B":
        weekend_mask = work[date_col].dt.dayofweek >= 5
        weekend_dropped = int(weekend_mask.sum())
        if weekend_dropped:
            work = work.loc[~weekend_mask]
        if work.empty:
            raise ValueError("All rows were removed during weekend filtering")

    start = work[date_col].min().date()
    end = work[date_col].max().date()
    holiday_idx = pd.DatetimeIndex([])
    if freq_tag == "B":
        holiday_idx = _resolve_holidays(start, end, holiday_calendar, holidays)
        if not holiday_idx.empty:
            work_normalised = work[date_col].dt.normalize()
            holiday_normalised = holiday_idx.normalize()
            holiday_mask = work_normalised.isin(holiday_normalised)
            holiday_dropped = int(holiday_mask.sum())
            if holiday_dropped:
                work = work.loc[~holiday_mask]
            if work.empty:
                raise ValueError("All rows were removed during holiday filtering")

    start_ts = work[date_col].min()
    end_ts = work[date_col].max()

    if freq_tag == "B":
        holidays_list = holiday_idx.to_pydatetime().tolist()
        business_offset = CustomBusinessDay(holidays=holidays_list)
        target_index = pd.date_range(start=start_ts, end=end_ts, freq=business_offset)
    else:
        target_index = pd.date_range(start=start_ts, end=end_ts, freq=freq_tag)

    if target_index.empty:
        target_index = pd.DatetimeIndex([start_ts])

    aligned = work.set_index(date_col).reindex(target_index)
    aligned.index.name = date_col
    aligned = aligned.reset_index().rename(columns={"index": date_col})

    info = {
        "target_frequency": freq_tag,
        "timezone": DEFAULT_TIMEZONE,
        "calendar": (holiday_calendar or DEFAULT_CALENDAR).lower(),
        "weekend_rows_dropped": weekend_dropped,
        "holiday_rows_dropped": holiday_dropped,
        "timestamp_count": len(target_index),
        "target_start": target_index[0].isoformat() if len(target_index) else None,
        "target_end": target_index[-1].isoformat() if len(target_index) else None,
    }
    attrs = dict(getattr(df, "attrs", {}))
    attrs["calendar_alignment"] = info
    aligned.attrs = attrs
    return aligned


__all__ = ["align_calendar"]
