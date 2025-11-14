"""Soft coverage tests for frequency utilities and time frequency helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_analysis.timefreq import (
    MONTHLY_DATE_FREQ,
    MONTHLY_PERIOD_FREQ,
    QUARTERLY_DATE_FREQ,
    QUARTERLY_PERIOD_FREQ,
    _validate_no_invalid_period_alias,
    assert_no_invalid_period_aliases_in_source,
    monthly_date_range,
    monthly_period_range,
)
from trend_analysis.util import frequency as freq_mod


def test_as_datetime_index_accepts_iterables_sorted() -> None:
    idx = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-03", "2024-01-02"], dtype="datetime64[ns]"
    )
    converted = freq_mod._as_datetime_index(idx)
    assert list(converted) == sorted(idx)

    converted_from_list = freq_mod._as_datetime_index(
        ["2024-02-01", "2024-02-03", "2024-02-02"]
    )
    assert list(converted_from_list) == sorted(converted_from_list)


def test_as_datetime_index_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="datetime-like"):
        freq_mod._as_datetime_index(["2024-01-01", "not-a-date"])  # type: ignore[arg-type]


def test_map_inferred_handles_known_aliases() -> None:
    assert freq_mod._map_inferred("W-MON") == "W"
    assert freq_mod._map_inferred("B") == "D"
    assert freq_mod._map_inferred("BM") == "M"
    assert freq_mod._map_inferred("Q-DEC") == "Q"
    assert freq_mod._map_inferred("A") == "Y"
    assert freq_mod._map_inferred("unknown") is None


def test_interval_classification_recognises_regular_cadence() -> None:
    base = pd.date_range("2024-01-01", periods=5, freq="D")
    diffs = freq_mod._intervals_in_days(base)
    assert np.allclose(diffs, 1.0)
    assert freq_mod._classify_from_diffs(diffs) == "D"

    weekly = pd.date_range("2024-01-01", periods=4, freq="W")
    weekly_diffs = freq_mod._intervals_in_days(weekly)
    assert freq_mod._classify_from_diffs(weekly_diffs) == "W"


def test_classify_from_diffs_rejects_irregular_series() -> None:
    diffs = np.array([1.0, 12.0, 200.0])
    with pytest.raises(ValueError, match="irregular"):
        freq_mod._classify_from_diffs(diffs)


def test_detect_frequency_uses_infer_freq_and_fallbacks() -> None:
    monthly = pd.date_range("2024-01-31", periods=6, freq="M")
    summary = freq_mod.detect_frequency(monthly)
    assert summary.code == "M"
    assert summary.label == "Monthly"
    assert summary.resampled is False

    irregular = pd.to_datetime(["2024-01-31", "2024-02-15", "2024-04-30"])
    with pytest.raises(ValueError):
        freq_mod.detect_frequency(irregular)


def test_detect_frequency_short_series_defaults_to_monthly() -> None:
    summary = freq_mod.detect_frequency(["2024-01-01"])
    assert summary.code == "M"


def test_summary_from_code_labels_consistently() -> None:
    summary = freq_mod._summary_from_code("W")
    assert summary.code == "W"
    assert summary.resampled is True
    assert summary.target == "M"


def test_monthly_date_and_period_ranges_follow_constants() -> None:
    dates = monthly_date_range("2024-01-31", periods=2)
    assert dates.freqstr == MONTHLY_DATE_FREQ
    periods = monthly_period_range("2024-01", periods=2)
    assert str(periods.freqstr) == MONTHLY_PERIOD_FREQ


def test_validate_no_invalid_period_alias_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        _validate_no_invalid_period_alias(MONTHLY_DATE_FREQ)

    # Valid period aliases do not raise
    _validate_no_invalid_period_alias(MONTHLY_PERIOD_FREQ)
    _validate_no_invalid_period_alias(QUARTERLY_PERIOD_FREQ)


def test_assert_no_invalid_period_aliases_in_source(tmp_path: Path) -> None:
    good = tmp_path / "ok.py"
    good.write_text("pd.period_range(start='2024-01', periods=2, freq='M')\n")

    bad = tmp_path / "bad.py"
    bad.write_text(
        'import pandas as pd\n' 'pd.period_range(start="2024-01", periods=2, freq="ME")\n'
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_no_invalid_period_aliases_in_source([str(good), str(bad)])

    assert "bad.py" in str(excinfo.value)


def test_frequency_summary_attributes_round_trip() -> None:
    summary = freq_mod.FrequencySummary("D", "Daily", False, "D", "Daily")
    assert summary.label == "Daily"
    assert summary.target_label == "Daily"

