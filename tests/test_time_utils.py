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


def test_align_calendar_normalises_timezones_and_infers_frequency():
    df = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2023-01-10 12:00", tz="US/Eastern"),
                pd.Timestamp("2023-01-11 12:00", tz="US/Eastern"),
            ],
            "value": [1, 2],
        }
    )

    result = align_calendar(df, frequency=None, timezone="US/Eastern")

    expected = [pd.Timestamp("2023-01-10 17:00"), pd.Timestamp("2023-01-11 17:00")]
    assert result["Date"].tolist() == expected
    assert result.attrs["calendar_alignment"]["target_frequency"] == "B"
    assert result.attrs["calendar_alignment"]["timezone"] == "UTC"


def test_align_calendar_defaults_to_simple_calendar_when_unknown():
    df = pd.DataFrame(
        {
            "Date": [
                dt.datetime(2023, 7, 3),
                dt.datetime(2023, 7, 4),
                dt.datetime(2023, 7, 5),
            ],
            "value": [10, 20, 30],
        }
    )

    result = align_calendar(df, holiday_calendar="unknown")

    assert result["Date"].dt.normalize().tolist() == [
        pd.Timestamp("2023-07-03"),
        pd.Timestamp("2023-07-05"),
    ]
    alignment = result.attrs["calendar_alignment"]
    assert alignment["calendar"] == "simple"
    assert alignment["holiday_rows_dropped"] == 1


def test_align_calendar_missing_column_raises_key_error():
    df = pd.DataFrame({"wrong": [pd.Timestamp("2023-01-01")]})

    with pytest.raises(KeyError, match="DataFrame must contain a 'Date' column"):
        align_calendar(df)


def test_align_calendar_all_invalid_dates_raise_value_error():
    df = pd.DataFrame({"Date": ["not-a-date", "also-bad"]})

    with pytest.raises(ValueError, match="Column 'Date' contains no valid timestamps"):
        align_calendar(df)


def test_align_calendar_weekly_frequency_retains_weekend_rows():
    df = pd.DataFrame(
        {
            "Date": [
                dt.datetime(2023, 7, 8),  # Saturday
                dt.datetime(2023, 7, 9),  # Sunday
                dt.datetime(2023, 7, 10),
            ],
            "value": [1, 2, 3],
        }
    )

    result = align_calendar(df, frequency="W")

    assert result["Date"].dt.normalize().tolist() == [pd.Timestamp("2023-07-08")]
    alignment = result.attrs["calendar_alignment"]
    assert alignment["weekend_rows_dropped"] == 0
    assert alignment["target_frequency"] == "W-FRI"


def test_align_calendar_holiday_overrides_are_range_filtered():
    df = pd.DataFrame(
        {
            "Date": [dt.datetime(2023, 7, 3), dt.datetime(2023, 7, 6)],
            "value": [10, 20],
        }
    )

    holidays = [
        pd.Timestamp("2022-12-25"),
        pd.Timestamp("2023-07-04"),
        pd.Timestamp("2024-01-01"),
    ]

    result = align_calendar(df, frequency="B", holidays=holidays)

    assert result["Date"].dt.normalize().tolist() == [
        pd.Timestamp("2023-07-03"),
        pd.Timestamp("2023-07-05"),
        pd.Timestamp("2023-07-06"),
    ]
    assert result.attrs["calendar_alignment"]["holiday_rows_dropped"] == 0
