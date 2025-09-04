"""Generate (in-sample, out-sample) period tuples for the multi-period engine.

Uses the new pandas offset aliases ``ME``/``QE``/``YE`` for month-, quarter-,
and year-end periods.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Dict, List, cast

import pandas as pd

# ----------------------------------------------------------------------
PeriodTuple = namedtuple("PeriodTuple", ["in_start", "in_end", "out_start", "out_end"])

FREQ_MAP = {
    # Standard codes + deprecated mappings
    "M": "ME",
    "ME": "ME",
    "Q": "QE",
    "QE": "QE",
    "A": "YE",
    "YE": "YE",
    # User-friendly names (all case variations)
    "monthly": "ME",
    "MONTHLY": "ME",
    "Monthly": "ME",
    "quarterly": "QE",
    "QUARTERLY": "QE",
    "Quarterly": "QE",
    "annual": "YE",
    "ANNUAL": "YE",
    "annually": "YE",
    "ANNUALLY": "YE",
    "Annually": "YE",
}


def generate_periods(cfg: Dict[str, Any]) -> List[PeriodTuple]:
    """Return a list of PeriodTuple driven by ``cfg["multi_period"]``.

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
