from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.tseries import offsets

__all__ = ["FrequencySummary", "FREQUENCY_LABELS", "detect_frequency"]

FrequencyCode = Literal["D", "W", "M"]

FREQUENCY_LABELS: dict[FrequencyCode, str] = {
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
}


@dataclass(frozen=True)
class FrequencySummary:
    """Summary of detected sampling cadence."""

    code: FrequencyCode
    label: str
    target: FrequencyCode
    resampled: bool

    @property
    def target_label(self) -> str:
        return FREQUENCY_LABELS[self.target]


def detect_frequency(index: pd.Index) -> FrequencySummary:
    """Return the sampling cadence for ``index``.

    Parameters
    ----------
    index:
        Datetime index describing the observations.

    Returns
    -------
    FrequencySummary
        Structured summary describing the detected frequency and whether the
        series needs to be resampled to the monthly cadence expected by the
        modelling pipeline.

    Raises
    ------
    TypeError
        If ``index`` is not a :class:`~pandas.DatetimeIndex`.
    ValueError
        When the frequency cannot be inferred with reasonable confidence.
    """

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("detect_frequency expects a DatetimeIndex")

    if len(index) < 2:
        raise ValueError("At least two timestamps are required to infer frequency")

    idx = index.sort_values().unique()

    inferred = pd.infer_freq(idx)
    if inferred:
        code = _map_offset_to_code(inferred)
        if code is not None:
            return FrequencySummary(code=code, label=FREQUENCY_LABELS[code], target="M", resampled=code != "M")

    code = _fallback_frequency_detection(idx)
    return FrequencySummary(code=code, label=FREQUENCY_LABELS[code], target="M", resampled=code != "M")


def _map_offset_to_code(freq: str) -> FrequencyCode | None:
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError):
        return None

    if isinstance(offset, (offsets.Day, offsets.BusinessDay, offsets.CDay, offsets.CustomBusinessDay)):
        return "D"
    if isinstance(offset, offsets.Week):
        return "W"
    if isinstance(
        offset,
        (
            offsets.MonthBegin,
            offsets.MonthEnd,
            offsets.BMonthBegin,
            offsets.BMonthEnd,
            offsets.SemiMonthBegin,
            offsets.SemiMonthEnd,
        ),
    ):
        return "M"
    return None


def _fallback_frequency_detection(index: pd.DatetimeIndex) -> FrequencyCode:
    if _looks_monthly(index):
        return "M"
    if _looks_weekly(index):
        return "W"
    if _looks_daily(index):
        return "D"
    raise ValueError("Unable to determine sampling frequency from index gaps")


def _looks_monthly(index: pd.DatetimeIndex) -> bool:
    months = index.to_period("M")
    if months.nunique() != len(index):
        return False
    deltas = np.diff(index.values.astype("datetime64[D]").astype(np.int64))
    if len(deltas) == 0:
        return True
    return float(np.mean((deltas >= 25) & (deltas <= 35))) >= 0.6


def _looks_weekly(index: pd.DatetimeIndex) -> bool:
    weeks = index.to_period("W")
    if weeks.nunique() != len(index):
        return False
    deltas = np.diff(index.values.astype("datetime64[D]").astype(np.int64))
    if len(deltas) == 0:
        return True
    ratio = float(np.mean((deltas >= 5) & (deltas <= 9)))
    return ratio >= 0.6


def _looks_daily(index: pd.DatetimeIndex) -> bool:
    deltas = np.diff(index.values.astype("datetime64[D]").astype(np.int64))
    if len(deltas) == 0:
        return True
    ratio = float(np.mean(deltas <= 4))
    return ratio >= 0.75
