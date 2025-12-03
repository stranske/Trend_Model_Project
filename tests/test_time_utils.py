import pandas as pd
import pytest

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
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03, 0.04],
        }
    )

    aligned = align_calendar(df, frequency="M")
    expected = df["Date"].dt.to_period("M").dt.to_timestamp()
    expected = expected + pd.offsets.BusinessMonthEnd()
    pd.testing.assert_series_equal(aligned["Date"], expected, check_names=False)
    meta = aligned.attrs.get("calendar_alignment", {})
    assert meta.get("target_frequency") == "M"
    assert meta.get("weekend_rows_dropped", 0) == 0
    assert meta.get("holiday_rows_dropped", 0) == 0


def test_align_calendar_respects_business_month_end_dates():
    dates = pd.to_datetime(["2024-06-28", "2024-07-31", "2024-08-30"])
    df = pd.DataFrame(
        {
            "Date": dates,
            "FundA": [0.01, 0.02, 0.03],
        }
    )

    aligned = align_calendar(df, frequency="ME")
    assert aligned["FundA"].isna().sum() == 0
    pd.testing.assert_index_equal(
        pd.Index(aligned["Date"]), pd.Index(dates), check_names=False
    )


def test_align_calendar_infers_business_frequency_from_data():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame({"Date": dates, "FundA": [1, 2, 3]})

    aligned = align_calendar(df, frequency=None)

    meta = aligned.attrs.get("calendar_alignment", {})
    assert meta.get("target_frequency") == "B"
    assert meta.get("timestamp_count") == len(dates)


def test_align_calendar_respects_holiday_overrides():
    dates = pd.to_datetime(["2024-12-23", "2024-12-24", "2024-12-25"])
    df = pd.DataFrame({"Date": dates, "FundA": [0.01, 0.02, 0.03]})

    override = [pd.Timestamp("2024-12-24")]
    aligned = align_calendar(df, holidays=override, holiday_calendar="nyse")

    assert pd.Timestamp("2024-12-24") not in set(aligned["Date"])
    meta = aligned.attrs.get("calendar_alignment", {})
    assert meta.get("holiday_rows_dropped", 0) == 1


def test_align_calendar_raises_when_all_weekends_removed():
    dates = pd.to_datetime(["2024-01-06", "2024-01-07"])  # weekend days
    df = pd.DataFrame({"Date": dates, "FundA": [0.1, 0.2]})

    with pytest.raises(ValueError, match="weekend"):
        align_calendar(df, frequency="B")


def test_align_calendar_handles_empty_dataframe():
    df = pd.DataFrame({"Date": pd.Series(dtype="datetime64[ns]"), "FundA": []})

    aligned = align_calendar(df, frequency="B")

    pd.testing.assert_frame_equal(aligned, df)
    meta = aligned.attrs.get("calendar_alignment", {})
    assert meta.get("timestamp_count") == 0
    assert meta.get("calendar") == "simple"


def test_align_calendar_rejects_invalid_dates():
    df = pd.DataFrame({"Date": ["not-a-date", "also-bad"], "FundA": [1, 2]})

    with pytest.raises(ValueError, match="no valid timestamps"):
        align_calendar(df, frequency="D")
