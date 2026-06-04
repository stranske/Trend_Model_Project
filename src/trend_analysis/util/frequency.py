"""Frequency-detection helpers used by the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "FrequencyCode",
    "FrequencySummary",
    "FREQUENCY_LABELS",
    "detect_frequency",
    "infer_periods_per_year",
]

FrequencyCode = Literal["D", "W", "M", "Q", "Y"]
FREQUENCY_LABELS: dict[FrequencyCode, str] = {
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
    "Q": "Quarterly",
    "Y": "Annual",
}


@dataclass(frozen=True, slots=True)
class FrequencySummary:
    """Structured result produced by :func:`detect_frequency`."""

    code: FrequencyCode
    label: str
    resampled: bool
    target: FrequencyCode
    target_label: str


def _as_datetime_index(index: Iterable[object]) -> pd.DatetimeIndex:
    """Return a normalised :class:`~pandas.DatetimeIndex` from ``index``."""

    if isinstance(index, pd.DatetimeIndex):
        return index.sort_values()
    try:
        idx = pd.DatetimeIndex(index)
    except (TypeError, ValueError):
        values = list(index)
        idx = pd.to_datetime(values, errors="coerce")
        mask = pd.isna(idx)
        if np.asarray(mask).any():
            raise ValueError("detect_frequency requires datetime-like inputs")
        idx = pd.DatetimeIndex(idx)
    return idx.sort_values()


def _map_inferred(freq: str | None) -> FrequencyCode | None:
    if not freq:
        return None
    freq = freq.upper()
    if freq.startswith("W"):
        return "W"
    if freq.endswith("D") or freq in {"B", "C", "BD"}:
        return "D"
    if any(freq.startswith(prefix) for prefix in ("M", "SM", "BM")):
        return "M"
    if freq.startswith("Q"):
        return "Q"
    if any(freq.startswith(prefix) for prefix in ("A", "Y")):
        return "Y"
    return None


def _intervals_in_days(idx: pd.DatetimeIndex) -> NDArray[np.float64]:
    deltas = np.diff(idx.to_numpy())
    return np.asarray(deltas / np.timedelta64(1, "D"), dtype=np.float64)


def _classify_from_diffs(diffs_days: NDArray[np.float64]) -> FrequencyCode:
    if diffs_days.size == 0:
        return "M"

    daily = (diffs_days > 0) & (diffs_days <= 4.0)
    weekly = (diffs_days >= 4.5) & (diffs_days <= 9.0)
    monthly = (diffs_days > 9.0) & (diffs_days <= 45.0)
    quarterly = (diffs_days > 45.0) & (diffs_days <= 120.0)
    yearly = diffs_days > 120.0

    buckets = {
        "D": int(daily.sum()),
        "W": int(weekly.sum()),
        "M": int(monthly.sum()),
        "Q": int(quarterly.sum()),
        "Y": int(yearly.sum()),
    }

    best_code = max(buckets, key=lambda code: buckets[code])
    best_count = buckets[best_code]
    total = diffs_days.size

    if best_count == 0:
        raise ValueError("Unable to determine series frequency from irregular spacing")

    if total >= 2 and (best_count / total) < 0.6:
        raise ValueError("Series cadence is too irregular to classify confidently")

    return cast(FrequencyCode, best_code)


def _summary_from_code(code: FrequencyCode) -> FrequencySummary:
    target: FrequencyCode = "M"
    resampled = code != target
    target_label = FREQUENCY_LABELS[target]
    label = FREQUENCY_LABELS[code]
    return FrequencySummary(
        code=code,
        label=label,
        resampled=resampled,
        target=target,
        target_label=target_label if resampled else label,
    )


def detect_frequency(index: Iterable[object]) -> FrequencySummary:
    """Classify a date index as daily, weekly or monthly.

    Parameters
    ----------
    index:
        Iterable of datetime-like values.  The iterable is converted to a
        :class:`~pandas.DatetimeIndex` internally; duplicates are ignored.

    Returns
    -------
    FrequencySummary
        Summary describing the detected cadence together with human-readable
        labelling and whether the series should be resampled to monthly.
    """

    idx = _as_datetime_index(index).drop_duplicates()
    if len(idx) < 2:
        return _summary_from_code("M")

    try:
        inferred = pd.infer_freq(idx)
    except ValueError:
        inferred = None

    detected = _map_inferred(inferred)
    if detected is not None:
        return _summary_from_code(detected)

    diffs_days = _intervals_in_days(idx)
    code = _classify_from_diffs(diffs_days)
    return _summary_from_code(code)


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer annualization periods from a datetime cadence."""

    if len(index) < 2:
        return 1

    diffs = np.diff(index.values.astype("datetime64[ns]").astype(np.int64))
    if len(diffs) == 0:
        return 1

    median_ns = np.median(diffs)
    if median_ns <= 0:
        return 1

    median_days = median_ns / (24 * 60 * 60 * 1e9)
    if median_days <= 0:
        return 1

    approx = int(round(365 / median_days))
    if approx >= 300:
        return 252
    if 45 <= approx <= 60:
        return 52
    if 10 <= approx <= 14:
        return 12
    if 3 <= approx <= 5:
        return 4
    return max(1, approx)
