import pandas as pd

from trend_analysis.util.frequency import FREQUENCY_LABELS, detect_frequency
from trend_analysis.util.missing import apply_missing_policy


def test_detect_frequency_daily_with_gaps():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    idx = idx.delete([2, 5])  # simulate holidays
    summary = detect_frequency(idx)
    assert summary.code == "D"
    assert summary.label == FREQUENCY_LABELS["D"]
    assert summary.resampled is True
    assert summary.target == "M"
    assert summary.target_label == FREQUENCY_LABELS["M"]


def test_detect_frequency_weekly():
    idx = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    summary = detect_frequency(idx)
    assert summary.code == "W"
    assert summary.label == FREQUENCY_LABELS["W"]
    assert summary.resampled is True
    assert summary.target == "M"
    assert summary.target_label == FREQUENCY_LABELS["M"]


def test_detect_frequency_monthly():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"])
    summary = detect_frequency(idx)
    assert summary.code == "M"
    assert summary.label == FREQUENCY_LABELS["M"]
    assert summary.resampled is False
    assert summary.target == "M"
    assert summary.target_label == FREQUENCY_LABELS["M"]


def test_detect_frequency_quarterly():
    idx = pd.date_range("2023-03-31", periods=5, freq="QE")
    summary = detect_frequency(idx)
    assert summary.code == "Q"
    assert summary.label == FREQUENCY_LABELS["Q"]
    assert summary.resampled is True
    assert summary.target == "M"
    assert summary.target_label == FREQUENCY_LABELS["M"]


def test_detect_frequency_annual():
    idx = pd.date_range("2018-12-31", periods=4, freq="YE")
    summary = detect_frequency(idx)
    assert summary.code == "Y"
    assert summary.label == FREQUENCY_LABELS["Y"]
    assert summary.resampled is True
    assert summary.target == "M"
    assert summary.target_label == FREQUENCY_LABELS["M"]


def test_apply_missing_policy_drop():
    df = pd.DataFrame(
        {
            "A": [1.0, float("nan"), 3.0],
            "B": [1.0, 2.0, 3.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    out, meta = apply_missing_policy(df, "drop", None)
    assert list(out.columns) == ["B"]
    assert meta["dropped"] == ["A"]


def test_apply_missing_policy_ffill_limit():
    df = pd.DataFrame(
        {
            "A": [1.0, float("nan"), float("nan"), 4.0],
            "B": [0.5, 0.6, 0.7, 0.8],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    out, meta = apply_missing_policy(df, "ffill", limit=1)
    assert "A" not in out.columns  # two consecutive NaNs exceed limit
    assert meta["dropped"] == ["A"]


def test_apply_missing_policy_zero_override():
    df = pd.DataFrame(
        {
            "A": [1.0, float("nan"), 3.0],
            "B": [0.5, float("nan"), 0.7],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    policy = {"default": "drop", "B": "zero"}
    out, meta = apply_missing_policy(df, policy, limit=None)
    assert "A" not in out.columns
    assert "B" in out.columns
    assert meta["filled"]["B"] == 1
    assert meta["policy"]["B"] == "zero"
