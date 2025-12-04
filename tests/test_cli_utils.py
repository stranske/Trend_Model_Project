import pandas as pd
import pytest

from trend_analysis import cli


def test_extract_cache_stats_returns_last_snapshot():
    payload = {
        "periods": [
            {"stats": {"entries": 1, "hits": 2, "misses": 3, "incremental_updates": 4}},
            {"stats": {"entries": 5.0, "hits": 6.0, "misses": 7.0, "incremental_updates": 8.0}},
        ],
        "irrelevant": [pd.Series([1, 2, 3])],
    }

    result = cli._extract_cache_stats(payload)

    assert result == {"entries": 5, "hits": 6, "misses": 7, "incremental_updates": 8}


def test_extract_cache_stats_walks_nested_sequences():
    payload = (
        [
            {"entries": 1, "hits": 1, "misses": 1, "incremental_updates": 1},
            [
                {"entries": 9.0, "hits": 10.0, "misses": 11.0, "incremental_updates": 12.0},
            ],
        ],
    )

    result = cli._extract_cache_stats(payload)

    assert result == {"entries": 9, "hits": 10, "misses": 11, "incremental_updates": 12}


def test_apply_universe_mask_with_missing_date_column_raises():
    df = pd.DataFrame({"A": [1, 2]})
    mask = pd.DataFrame(index=pd.RangeIndex(2), columns=["A"], data=True)

    with pytest.raises(KeyError):
        cli._apply_universe_mask(df, mask, date_column="date")


def test_apply_universe_mask_applies_membership_and_preserves_dates():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
        }
    )
    mask = pd.DataFrame(
        {
            "A": [True, False],
            "B": [True, True],
        },
        index=df["date"],
    )

    masked = cli._apply_universe_mask(df, mask, date_column="date")

    assert masked.loc[0, "A"] == 1.0
    assert pd.isna(masked.loc[1, "A"])
    assert masked.loc[1, "B"] == 4.0
    pd.testing.assert_series_equal(masked["date"], df["date"], check_names=False)


def test_apply_universe_mask_raises_for_missing_members():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "A": [1.0],
        }
    )
    mask = pd.DataFrame({"B": [True]}, index=df["date"])

    with pytest.raises(KeyError):
        cli._apply_universe_mask(df, mask, date_column="date")
