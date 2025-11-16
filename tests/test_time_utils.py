import pandas as pd

from trend_analysis.time_utils import align_calendar


def test_align_calendar_removes_weekends_and_holidays():
    dates = pd.to_datetime(
        [
            "2024-11-27",  # Wednesday
            "2024-11-28",  # Thanksgiving (holiday)
            "2024-11-29",  # Friday
            "2024-11-30",  # Saturday
            "2024-12-02",  # Monday
        ]
    )
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )

    aligned = align_calendar(df, frequency="D", timezone="US/Eastern")
    assert (aligned["Date"].dt.dayofweek < 5).all()
    assert pd.Timestamp("2024-11-28") not in set(aligned["Date"])  # holiday removed

    meta = aligned.attrs.get("calendar_alignment", {})
    assert meta.get("weekend_rows_dropped", 0) >= 1
    assert meta.get("holiday_rows_dropped", 0) >= 1


def test_align_calendar_preserves_monthly_frequency():
    dates = pd.date_range("2024-01-31", periods=4, freq="M")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03, 0.04],
        }
    )

    aligned = align_calendar(df, frequency="M")
    pd.testing.assert_series_equal(aligned["Date"], df["Date"], check_names=False)
    meta = aligned.attrs.get("calendar_alignment", {})
    assert meta.get("target_frequency") == "M"
    assert meta.get("weekend_rows_dropped", 0) == 0
    assert meta.get("holiday_rows_dropped", 0) == 0
