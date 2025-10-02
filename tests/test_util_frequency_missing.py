import pandas as pd
import pytest

from trend_analysis.util.frequency import FREQUENCY_LABELS, detect_frequency
from trend_analysis.util.missing import MissingPolicyResult, apply_missing_policy


def test_detect_frequency_daily_with_holiday_gap():
    dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",  # weekend gap
            "2024-01-09",
            "2024-01-11",  # simulate market holiday on the 10th
        ]
    )
    summary = detect_frequency(pd.DatetimeIndex(dates))
    assert summary.code == "D"
    assert summary.label == FREQUENCY_LABELS["D"]
    assert summary.resampled is True


def test_detect_frequency_weekly_series():
    dates = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    summary = detect_frequency(pd.DatetimeIndex(dates))
    assert summary.code == "W"
    assert summary.label == FREQUENCY_LABELS["W"]
    assert summary.resampled is True


def test_detect_frequency_monthly_series():
    dates = pd.date_range("2023-01-31", periods=6, freq="ME")
    summary = detect_frequency(pd.DatetimeIndex(dates))
    assert summary.code == "M"
    assert summary.label == FREQUENCY_LABELS["M"]
    assert summary.resampled is False


def test_detect_frequency_irregular_raises():
    dates = pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-01", "2024-02-05"])
    with pytest.raises(ValueError):
        detect_frequency(pd.DatetimeIndex(dates))


def test_apply_missing_policy_drop_removes_assets():
    frame = pd.DataFrame(
        {
            "A": [0.01, 0.02, None, 0.03],
            "B": [0.01, 0.01, 0.01, 0.01],
        },
        index=pd.date_range("2024-01-31", periods=4, freq="ME"),
    )
    cleaned, result = apply_missing_policy(frame, "drop")
    assert list(cleaned.columns) == ["B"]
    assert isinstance(result, MissingPolicyResult)
    assert result.dropped_assets == ("A",)
    assert result.total_filled == 0


def test_apply_missing_policy_ffill_with_limit():
    frame = pd.DataFrame(
        {
            "A": [0.01, None, None, 0.02],
        },
        index=pd.date_range("2024-01-31", periods=4, freq="ME"),
    )
    cleaned, result = apply_missing_policy(frame, "ffill", limit=1)
    assert cleaned.iloc[1, 0] == pytest.approx(0.01)
    # limit prevents the second consecutive NaN from filling
    assert pd.isna(cleaned.iloc[2, 0])
    assert result.filled_cells == (("A", 1),)


def test_apply_missing_policy_zero_fill():
    frame = pd.DataFrame(
        {
            "A": [0.01, None, 0.03],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )
    cleaned, result = apply_missing_policy(frame, "zero")
    assert cleaned.iloc[1, 0] == 0.0
    assert result.total_filled == 1
