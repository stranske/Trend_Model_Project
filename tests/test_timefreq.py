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


def test_assert_no_invalid_period_aliases_skips_known_false_positives(
    tmp_path: Path,
) -> None:
    """Files under virtualenvs or this module itself are intentionally
    ignored."""

    skipped_env = tmp_path / ".venv" / "lib" / "python.py"
    skipped_env.parent.mkdir(parents=True)
    skipped_env.write_text(
        "pd.period_range('2024-01', periods=1, freq=\"ME\")\n",
        encoding="utf-8",
    )

    skipped_module = tmp_path / "timefreq.py"
    skipped_module.write_text(
        "pd.period_range('2024-01', periods=1, freq=\"QE\")\n",
        encoding="utf-8",
    )

    good_file = tmp_path / "good.py"
    good_file.write_text(
        "pd.period_range('2024-01', periods=1, freq=\"M\")\n",
        encoding="utf-8",
    )

    # Even though the skipped files contain invalid aliases, they should be ignored.
    timefreq.assert_no_invalid_period_aliases_in_source(
        [str(skipped_env), str(skipped_module), str(good_file)]
    )


def test_assert_no_invalid_period_aliases_handles_unreadable_files(
    tmp_path: Path,
) -> None:
    unreadable_file = tmp_path / "missing.py"
    good_file = tmp_path / "good.py"
    good_file.write_text(
        "pd.period_range('2024-01', periods=1, freq=\"M\")\n",
        encoding="utf-8",
    )

    # The helper should silently skip files that cannot be opened, relying on
    # the caller to provide only project-controlled paths.
    timefreq.assert_no_invalid_period_aliases_in_source([str(unreadable_file), str(good_file)])


def test_validate_no_invalid_period_alias_includes_helpful_message() -> None:
    with pytest.raises(ValueError) as excinfo:
        timefreq._validate_no_invalid_period_alias(timefreq.MONTHLY_DATE_FREQ)

    message = str(excinfo.value)
    assert timefreq.MONTHLY_PERIOD_FREQ in message
    assert timefreq.QUARTERLY_PERIOD_FREQ in message
