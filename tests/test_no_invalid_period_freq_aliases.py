"""Guardrail test: prevent accidental use of ME/QE with pd.period_range.

This avoids a class of regressions where a broad search/replace (M->ME)
breaks period-based tests by introducing ValueError at runtime.
"""

from __future__ import annotations

from pathlib import Path
from trend_analysis.timefreq import assert_no_invalid_period_aliases_in_source


def test_no_invalid_period_freq_aliases() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [str(p) for p in root.rglob("*.py")]
    assert_no_invalid_period_aliases_in_source(paths)


def test_period_and_date_range_examples() -> None:
    # Sanity check the documented behaviour
    import pandas as pd
    from trend_analysis import timefreq as tf

    ts_idx = pd.date_range("2024-01-31", periods=2, freq=tf.MONTHLY_DATE_FREQ)
    assert len(ts_idx) == 2
    pr_idx = pd.period_range("2024-01", periods=2, freq=tf.MONTHLY_PERIOD_FREQ)
    assert len(pr_idx) == 2
    # Intentionally do NOT attempt pd.period_range(..., freq=ME) here; that is
    # what the guard test above forbids statically.
