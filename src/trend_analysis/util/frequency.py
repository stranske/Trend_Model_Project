"""Frequency-detection helpers used by the analysis pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Iterable as TypingIterable
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["detect_frequency", "FrequencyCode"]

FrequencyCode = Literal["D", "W", "M"]


def _as_datetime_index(index: Iterable[object]) -> pd.DatetimeIndex:
    """Return a normalised :class:`~pandas.DatetimeIndex` from ``index``."""

    if isinstance(index, pd.DatetimeIndex):
        return index.sort_values()
    try:
        idx = pd.DatetimeIndex(index)
    except (TypeError, ValueError):
        idx = pd.to_datetime(list(index), errors="coerce")
        if getattr(idx, "isna", lambda: True)().any():
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
    if any(freq.startswith(prefix) for prefix in ("Q", "A", "Y")):
        return "M"
    return None


def detect_frequency(index: TypingIterable[object]) -> FrequencyCode:
    """Classify a date index as daily, weekly or monthly.

    Parameters
    ----------
    index:
        Iterable of datetime-like values.  The iterable is converted to a
        :class:`~pandas.DatetimeIndex` internally; duplicates are ignored.

    Returns
    -------
    Literal["D", "W", "M"]
        Canonicalised frequency code representing daily, weekly or monthly
        cadence.  The detector tolerates gaps introduced by market holidays and
        short calendar months by relying on the median interval once
        :func:`pandas.infer_freq` cannot produce a definitive answer.
    """

    idx = _as_datetime_index(index).drop_duplicates()
    if len(idx) < 2:
        return "M"

    detected = _map_inferred(pd.infer_freq(idx))
    if detected is not None:
        return detected

    diffs = np.diff(idx.view("i8"))  # nanoseconds
    if diffs.size == 0:
        return "M"
    diffs_days = diffs / 86_400_000_000_000  # ns -> days
    median = float(np.median(diffs_days))
    if median <= 3.5:
        return "D"
    if median <= 12:
        return "W"
    return "M"

