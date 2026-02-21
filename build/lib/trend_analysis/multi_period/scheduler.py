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

    start_mode = str(mp.get("start_mode", "in") or "in").lower()
    start_is_oos = start_mode in {"oos", "out", "out_sample", "out-of-sample"}

    periods: List[PeriodTuple] = []

    # Work in month periods to avoid off-by-one month-end math.
    end_period = end_date.to_period("M")

    if start_is_oos:
        out_start_period = start_date.to_period("M")
    else:
        in_start_period = start_date.to_period("M")
        out_start_period = None

    while True:
        if start_is_oos:
            in_end_period = out_start_period - 1
            in_months = step_months * in_len
            in_start_period = in_end_period - (in_months - 1)
        else:
            in_months = step_months * in_len
            in_end_period = in_start_period + (in_months - 1)
            out_start_period = in_end_period + 1

        out_months = step_months * out_len
        out_end_period = out_start_period + (out_months - 1)

        # Stop if we would have no out-of-sample data at all.
        if out_start_period > end_period:
            break

        # Allow a truncated final out-of-sample window.
        # This supports partial final periods (e.g., fewer than 12 months for
        # annual frequency) without annualising/normalising to a full window.
        is_final_partial = out_end_period > end_period
        if is_final_partial:
            out_end_period = end_period

        in_start = in_start_period.to_timestamp("M", how="end")
        in_end = in_end_period.to_timestamp("M", how="end")
        out_start = out_start_period.to_timestamp("M", how="end")
        out_end = out_end_period.to_timestamp("M", how="end")

        periods.append(
            PeriodTuple(
                in_start=str(in_start.date()),
                in_end=str(in_end.date()),
                out_start=str(out_start.date()),
                out_end=str(out_end.date()),
            )
        )

        if is_final_partial:
            break

        # Advance by out_len periods (advance OOS window start).
        out_start_period = out_start_period + (step_months * out_len)
        if not start_is_oos:
            in_start_period = in_start_period + (step_months * out_len)

    return periods
