"""Unit tests for :mod:`trend_analysis.timefreq`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend_analysis import timefreq


def test_monthly_date_range_uses_policy_frequency() -> None:
    idx = timefreq.monthly_date_range("2024-01-31", periods=3)
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.freqstr == timefreq.MONTHLY_DATE_FREQ
    assert list(idx) == list(
        pd.date_range("2024-01-31", periods=3, freq=timefreq.MONTHLY_DATE_FREQ)
    )


def test_monthly_period_range_uses_policy_frequency() -> None:
    idx = timefreq.monthly_period_range("2024-01", periods=2)
    assert isinstance(idx, pd.PeriodIndex)
    assert idx.freqstr == timefreq.MONTHLY_PERIOD_FREQ
    assert list(idx.astype(str)) == ["2024-01", "2024-02"]


def test_validate_no_invalid_period_alias_rejects_timestamp_aliases() -> None:
    timefreq._validate_no_invalid_period_alias(timefreq.MONTHLY_PERIOD_FREQ)
    timefreq._validate_no_invalid_period_alias(timefreq.QUARTERLY_PERIOD_FREQ)
    with pytest.raises(ValueError):
        timefreq._validate_no_invalid_period_alias(timefreq.MONTHLY_DATE_FREQ)
    with pytest.raises(ValueError):
        timefreq._validate_no_invalid_period_alias(timefreq.QUARTERLY_DATE_FREQ)


def test_assert_no_invalid_period_aliases_in_source(tmp_path: Path) -> None:
    good_file = tmp_path / "good.py"
    good_file.write_text(
        "pd.period_range('2024-01', periods=1, freq=\"M\")\n",
        encoding="utf-8",
    )

    bad_file = tmp_path / "bad.py"
    bad_file.write_text(
        "pd.period_range('2024-01', periods=1, freq=\"ME\")\n",
        encoding="utf-8",
    )

    timefreq.assert_no_invalid_period_aliases_in_source([str(good_file)])

    with pytest.raises(AssertionError) as excinfo:
        timefreq.assert_no_invalid_period_aliases_in_source([str(bad_file)])

    message = str(excinfo.value)
    assert "Invalid period_range frequency alias detected" in message
    assert str(bad_file) in message
