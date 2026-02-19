"""Centralised date/time frequency aliases and helpers.

Policy (see docs/datetime-frequency-policy.md):
    * For timestamp indices (`pd.date_range`) use new explicit end-based
      aliases: monthly = "ME", quarterly = "QE".
    * For period indices (`pd.period_range`) Pandas (as of Sept 2025) still
      requires legacy aliases: monthly = "M", quarterly = "Q". Attempting
      to use "ME" / "QE" with `period_range` raises ``ValueError``.

Rationale:
    Minimises FutureWarning noise for timestamp ranges while avoiding
    ValueError for period-based windows. This module provides constants so
    call sites can be self-documenting and reduces future churn if Pandas
    harmonises period aliases later.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

__all__ = [
    "MONTHLY_DATE_FREQ",
    "MONTHLY_PERIOD_FREQ",
    "QUARTERLY_DATE_FREQ",
    "QUARTERLY_PERIOD_FREQ",
    "monthly_date_range",
    "monthly_period_range",
]

# Public constants (do not inline magic strings elsewhere)
MONTHLY_DATE_FREQ = "ME"  # month-end (Timestamp index)
MONTHLY_PERIOD_FREQ = "M"  # Period index monthly
QUARTERLY_DATE_FREQ = "QE"  # quarter-end (Timestamp index)
QUARTERLY_PERIOD_FREQ = "Q"  # Period index quarterly


def monthly_date_range(
    start: str | date, periods: int | None = None, end: str | date | None = None
) -> pd.DatetimeIndex:
    """Wrapper around ``pd.date_range`` using the policy monthly date freq.

    Exactly one of (``periods``, ``end``) must be supplied (delegated to pandas).
    """
    return pd.date_range(start=start, periods=periods, end=end, freq=MONTHLY_DATE_FREQ)


def monthly_period_range(
    start: str | date, periods: int | None = None, end: str | date | None = None
) -> pd.PeriodIndex:
    """Wrapper around ``pd.period_range`` using the policy monthly period freq.

    Uses legacy alias 'M' intentionally (see module docstring).
    """
    return pd.period_range(start=start, periods=periods, end=end, freq=MONTHLY_PERIOD_FREQ)


def _validate_no_invalid_period_alias(freq: str) -> None:
    """Internal guard used by tests; raises if an invalid period alias is
    passed.

    Provided as a small reusable assertion should runtime validation
    ever be required at ingestion boundaries.
    """
    if freq in {MONTHLY_DATE_FREQ, QUARTERLY_DATE_FREQ}:  # ME or QE
        raise ValueError(
            f"Invalid period frequency '{freq}'. Use '{MONTHLY_PERIOD_FREQ}' or "
            f"'{QUARTERLY_PERIOD_FREQ}' for period ranges."
        )


def assert_no_invalid_period_aliases_in_source(paths: Iterable[str]) -> None:
    """Scan given repository python files for forbidden ``period_range`` usage.

    We purposely DO NOT traverse virtualenv / site-packages to avoid
    matching upstream library test suites. The caller should pass only
    project-controlled files (tests + src). The module docstring here is
    ignored by requiring ``freq="`` to appear on the same logical line as
    the call (preventing broad descriptive text from matching).
    """
    import re

    pattern = re.compile(r"period_range\([^\n]*freq=\"(ME|QE)\"")
    bad: list[tuple[str, str]] = []
    skip_tokens = {"/.venv/", "/.autofix-venv/", "/site-packages/"}

    for p in paths:
        if any(token in p for token in skip_tokens) or p.endswith(
            "timefreq.py"
        ):  # skip envs + this file’s examples
            continue
        try:
            text = open(p, "r", encoding="utf-8").read()
        except OSError:
            continue
        for m in pattern.finditer(text):
            snippet = text[m.start() : m.end()]
            bad.append((p, snippet))
    if bad:
        joined = "\n".join(f"{fp}: {snip}" for fp, snip in bad)
        raise AssertionError(
            "Invalid period_range frequency alias detected. Use 'M'/'Q' for Period indices.\n"
            + joined
        )
