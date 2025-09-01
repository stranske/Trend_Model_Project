"""Generate (in-sample, out-sample) period tuples for the multi-period engine.

Uses pandas offset aliases ``ME``/``Q``/``A`` for month-, quarter-, and
year-end periods. Legacy aliases like ``M`` are mapped to ``ME`` for
backward compatibility.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Dict, List, cast

import pandas as pd

# ----------------------------------------------------------------------
PeriodTuple = namedtuple("PeriodTuple", ["in_start", "in_end", "out_start", "out_end"])

FREQ_MAP = {
    # Standard codes + deprecated mappings
    "ME": "ME",
    "M": "ME",
    "Q": "Q",
    "QE": "Q",
    "A": "A",
    "YE": "A",
    # User-friendly names (all case variations)
    "monthly": "ME",
    "MONTHLY": "ME",
    "Monthly": "ME",
    "quarterly": "Q",
    "QUARTERLY": "Q",
    "Quarterly": "Q",
    "annual": "A",
    "ANNUAL": "A",
    "annually": "A",
    "ANNUALLY": "A",
    "Annually": "A",
}


def generate_periods(cfg: Dict[str, Any]) -> List[PeriodTuple]:
    """
    Return a list of PeriodTuple driven by ``cfg["multi_period"]``.

    • Clock jumps forward by the out‑of‑sample window length
    • In‑sample length = ``in_sample_len`` windows
    • Generation stops when the next OOS window would exceed ``end``.
    """
    mp = cast(Dict[str, Any], cfg.get("multi_period", {}))

    freq_alias = FREQ_MAP[str(mp["frequency"])]
    offset = pd.tseries.frequencies.to_offset(freq_alias)
    in_len = int(mp["in_sample_len"])
    out_len = int(mp["out_sample_len"])

    start = pd.Period(str(mp["start"]), offset)
    last = pd.Period(str(mp["end"]), offset)

    periods: List[PeriodTuple] = []
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
        in_start += out_len  # jump ahead by OOS length

    return periods
