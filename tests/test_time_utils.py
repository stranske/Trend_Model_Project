import datetime as dt

import pandas as pd
import pytest

from trend_analysis.time_utils import align_calendar


def test_align_calendar_empty_preserves_metadata():
    df = pd.DataFrame({"Date": []})

    result = align_calendar(df)

    assert result.empty
    assert result.attrs["calendar_alignment"] == {
        "target_frequency": "B",
        "timezone": "UTC",
        "calendar": "simple",
        "timestamp_count": 0,
    }


def test_weekend_only_raises_after_filtering():
    df = pd.DataFrame(
        {
            "Date": [
                dt.datetime(2023, 7, 8),  # Saturday
                dt.datetime(2023, 7, 9),  # Sunday
            ],
            "value": [1, 2],
        }
    )

    with pytest.raises(
        ValueError, match="All rows were removed during weekend filtering"
    ):
        align_calendar(df)


def test_holiday_overrides_drop_rows_and_record_metadata():
    df = pd.DataFrame(
        {
            "Date": [
                dt.datetime(2023, 7, 3),
                dt.datetime(2023, 7, 4),
            ],
            "value": [10, 20],
        }
    )

    holidays = [pd.Timestamp(dt.date(2023, 7, 4))]
    result = align_calendar(df, frequency="B", holidays=holidays)

    assert result["Date"].dt.normalize().tolist() == [pd.Timestamp("2023-07-03")]
    assert result["value"].tolist() == [10]
    assert result.attrs["calendar_alignment"]["holiday_rows_dropped"] == 1
    assert result.attrs["calendar_alignment"]["weekend_rows_dropped"] == 0


def test_monthly_frequency_expands_to_business_month_end():
    df = pd.DataFrame(
        {
            "Date": [dt.datetime(2023, 1, 1), dt.datetime(2023, 3, 15)],
            "value": [5, 7],
        }
    )

    result = align_calendar(df, frequency="M")

    expected_dates = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"])
    assert result["Date"].tolist() == list(expected_dates)
    assert (
        result.loc[result["Date"] == pd.Timestamp("2023-02-28"), "value"].isna().all()
    )
    assert result.attrs["calendar_alignment"]["target_frequency"] == "M"


def test_align_calendar_requires_date_column():
    df = pd.DataFrame({"value": [1, 2]})

    with pytest.raises(KeyError, match="DataFrame must contain a 'Date' column"):
        align_calendar(df)


def test_align_calendar_rejects_all_invalid_dates():
    df = pd.DataFrame({"Date": ["not-a-date", None], "value": [1, 2]})

    with pytest.raises(ValueError, match="contains no valid timestamps"):
        align_calendar(df)


def test_align_calendar_infers_weekly_frequency_without_weekend_drop():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2023-07-07", periods=2, freq="W-FRI"),
            "value": [1, 2],
        }
    )

    aligned = align_calendar(df, frequency="W")

    assert aligned["Date"].tolist() == df["Date"].tolist()
    assert aligned.attrs["calendar_alignment"]["target_frequency"] == "W-FRI"
    assert aligned.attrs["calendar_alignment"]["weekend_rows_dropped"] == 0
    assert aligned.attrs["calendar_alignment"]["holiday_rows_dropped"] == 0


def test_align_calendar_drops_weekends_and_default_holidays():
    df = pd.DataFrame(
        {
            "Date": [
                dt.datetime(2023, 6, 30),  # Friday
                dt.datetime(2023, 7, 1),  # Saturday
                dt.datetime(2023, 7, 4),  # Tuesday (holiday)
            ],
            "value": [1, 2, 3],
        }
    )

    aligned = align_calendar(df, frequency="B", holiday_calendar="simple")

    assert aligned["Date"].tolist() == [pd.Timestamp("2023-06-30")]
    assert aligned.attrs["calendar_alignment"]["weekend_rows_dropped"] == 1
    assert aligned.attrs["calendar_alignment"]["holiday_rows_dropped"] == 1


def test_align_calendar_converts_timezones_and_preserves_attrs():
    df = pd.DataFrame(
        {
            "Date": [dt.datetime(2023, 1, 2), dt.datetime(2023, 1, 3)],
            "value": [10, 20],
        }
    )
    df.attrs["original"] = "present"

    aligned = align_calendar(df, timezone="US/Eastern")

    expected_dates = [pd.Timestamp("2023-01-03 05:00:00")]
    assert aligned["Date"].tolist() == expected_dates
    assert aligned.attrs["original"] == "present"
    assert aligned.attrs["calendar_alignment"]["timezone"] == "UTC"
    assert aligned.attrs["calendar_alignment"]["holiday_rows_dropped"] == 1


def test_align_calendar_normalises_unknown_calendar_name():
    df = pd.DataFrame({"Date": [dt.datetime(2023, 2, 1)], "value": [1.0]})

    aligned = align_calendar(df, frequency="B", holiday_calendar="custom")

    assert aligned.attrs["calendar_alignment"]["calendar"] == "simple"


def test_align_calendar_honours_frequency_aliases_and_tags():
    df = pd.DataFrame({"Date": [dt.datetime(2023, 2, 1)], "value": [1.0]})

    daily = align_calendar(df, frequency="D")
    assert daily.attrs["calendar_alignment"]["target_frequency"] == "B"

    weekly = align_calendar(df, frequency="W-SUN")
    assert weekly.attrs["calendar_alignment"]["target_frequency"] == "W-SUN"

    quarterly = align_calendar(df, frequency="Q")
    assert quarterly.attrs["calendar_alignment"]["target_frequency"] == "Q"


def test_align_calendar_converts_timezone_aware_series():
    tz_index = pd.date_range("2023-03-01", periods=2, freq="D", tz="UTC")
    df = pd.DataFrame({"Date": tz_index, "value": [3, 4]})

    aligned = align_calendar(df, frequency="B", timezone="US/Eastern")

    assert aligned["Date"].dt.tz is None
    assert aligned.attrs["calendar_alignment"]["timezone"] == "UTC"


def test_align_calendar_handles_empty_holiday_index():
    df = pd.DataFrame({"Date": [dt.datetime(2023, 1, 5), dt.datetime(2023, 1, 6)], "value": [1, 2]})

    aligned = align_calendar(df, frequency="B", holiday_calendar="simple")

    assert aligned.attrs["calendar_alignment"]["holiday_rows_dropped"] == 0
    assert aligned["Date"].tolist() == pd.to_datetime(["2023-01-05", "2023-01-06"]).tolist()
