"""
Generate (in‑sample, out‑sample) period tuples for the multi‑period engine.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List, Mapping, Any
import pandas as pd

PeriodTuple = namedtuple(
    "PeriodTuple",
    ["in_start", "in_end", "out_start", "out_end"],
)
FREQ_MAP = {"M": "M", "Q": "Q", "A": "Y"}


def generate_periods(cfg: Mapping[str, Any]) -> List[PeriodTuple]:
    """
    Returns a list of PeriodTuple driven by ``cfg.multi_period``.

    If ``cfg`` lacks a ``multi_period`` section an empty list is returned.

    • Clock jumps forward by the *out‑of‑sample* window length.
    • In‑sample length = cfg.multi_period.in_sample_len windows.
    • Frequency ∈ {'M','Q','A'}.
    • Generation stops when the end of the next OOS window
      would run past cfg.multi_period.end.
    """
    mp = cfg.get("multi_period")
    if mp is None:
        return []
    freq = FREQ_MAP[mp["frequency"]]
    in_len = int(mp["in_sample_len"])
    out_len = int(mp["out_sample_len"])

    start = pd.Period(mp["start"], freq)
    last = pd.Period(mp["end"], freq)

    periods: list[PeriodTuple] = []
    in_start = start

    while True:
        in_end = in_start + in_len - 1
        out_start = in_end + 1
        out_end = out_start + out_len - 1
        if out_end > last:
            break

        periods.append(
            PeriodTuple(
                in_start=str(in_start.start_time.date()),
                in_end=str(in_end.end_time.date()),
                out_start=str(out_start.start_time.date()),
                out_end=str(out_end.end_time.date()),
            )
        )
        in_start = in_start + out_len  # jump ahead

    return periods
