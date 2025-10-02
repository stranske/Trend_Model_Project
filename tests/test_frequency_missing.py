import pandas as pd

from trend_analysis.util.frequency import detect_frequency
from trend_analysis.util.missing import apply_missing_policy


def test_detect_frequency_daily_with_gaps():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    idx = idx.delete([2, 5])  # simulate holidays
    assert detect_frequency(idx) == "D"


def test_detect_frequency_weekly():
    idx = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    assert detect_frequency(idx) == "W"


def test_detect_frequency_monthly():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"])
    assert detect_frequency(idx) == "M"


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
