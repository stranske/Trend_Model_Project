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

    For annual frequency, the start month from the config is preserved.
    For example, start="2003-07" with annual frequency will produce periods
    starting in July each year, not January.
    """
    mp = cast(Dict[str, Any], cfg.get("multi_period", {}))

    freq_str = str(mp["frequency"]).upper()
    in_len = int(mp["in_sample_len"])
    out_len = int(mp["out_sample_len"])

    # Parse start/end dates - preserve the actual month
    start_str = str(mp["start"])
    end_str = str(mp["end"])

    # Handle both "YYYY-MM" and "YYYY-MM-DD" formats
    if len(start_str) == 7:  # "YYYY-MM"
        start_date = pd.Timestamp(f"{start_str}-01") + pd.offsets.MonthEnd(0)
    else:
        start_date = pd.Timestamp(start_str)

    if len(end_str) == 7:  # "YYYY-MM"
        end_date = pd.Timestamp(f"{end_str}-01") + pd.offsets.MonthEnd(0)
    else:
        end_date = pd.Timestamp(end_str)

    # Determine step size based on frequency
    if freq_str in ("A", "YE", "ANNUAL", "ANNUALLY"):
        step_months = 12
    elif freq_str in ("Q", "QE", "QUARTERLY"):
        step_months = 3
    else:  # Monthly
        step_months = 1

    periods: List[PeriodTuple] = []

    # Start with in-sample period
    in_start = start_date

    while True:
        # In-sample end: in_len periods after in_start
        in_end = (
            in_start
            + pd.DateOffset(months=step_months * max(in_len - 1, 0))
            - pd.Timedelta(days=1)
        )
        # Align to month end
        in_end = in_end + pd.offsets.MonthEnd(0)

        # Out-sample start: day after in_end
        out_start = in_end + pd.Timedelta(days=1)

        # Out-sample end: out_len periods after out_start
        out_end = (
            out_start
            + pd.DateOffset(months=step_months * out_len)
            - pd.Timedelta(days=1)
        )
        out_end = out_end + pd.offsets.MonthEnd(0)

        # Stop if out-sample exceeds end date
        if out_end > end_date:
            break

        periods.append(
            PeriodTuple(
                in_start=str(in_start.date()),
                in_end=str(in_end.date()),
                out_start=str(out_start.date()),
                out_end=str(out_end.date()),
            )
        )

        # Advance by out_len periods
        in_start = in_start + pd.DateOffset(months=step_months * out_len)

    return periods
