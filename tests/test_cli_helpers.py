import numpy as np
import pandas as pd
import pytest

from trend_analysis import cli


def test_extract_cache_stats_returns_last_valid_snapshot():
    payload = {
        "early": {"entries": 1, "hits": 0, "misses": 1, "incremental_updates": 0},
        "nested": [
            {"entries": 2, "hits": 1, "misses": 1, "incremental_updates": 0.0},
            "ignore",
            {"entries": 3, "hits": 2, "misses": 0, "incremental_updates": 1},
        ],
        "frame": pd.DataFrame({"A": [1, 2]}),
        "array": np.array([1, 2, 3]),
    }

    stats = cli._extract_cache_stats(payload)

    assert stats == {"entries": 3, "hits": 2, "misses": 0, "incremental_updates": 1}


def test_extract_cache_stats_returns_none_when_absent():
    payload = {"no_stats": [{"entries": 1, "hits": 1}], "series": pd.Series([1.0])}

    assert cli._extract_cache_stats(payload) is None


def test_apply_universe_mask_filters_and_preserves_dates():
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
        }
    )
    mask = pd.DataFrame(
        {"A": [True, False], "B": [False, True]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )

    result = cli._apply_universe_mask(df, mask, date_column="date")

    assert list(result.columns) == ["Date", "A", "B"]
    assert pd.to_datetime(result["Date"]).tolist() == list(mask.index)
    assert np.isnan(result.loc[0, "B"]) and np.isnan(result.loc[1, "A"])
    assert result.loc[0, "A"] == 1.0 and result.loc[1, "B"] == 4.0


def test_apply_universe_mask_raises_for_missing_members():
    df = pd.DataFrame({"date": ["2020-01-01"], "A": [1.0]})
    mask = pd.DataFrame({"Missing": [True]}, index=pd.to_datetime(["2020-01-01"]))

    with pytest.raises(KeyError):
        cli._apply_universe_mask(df, mask, date_column="date")


def test_apply_universe_mask_short_circuits_on_empty_mask():
    df = pd.DataFrame({"date": ["2020-01-01"], "A": [1.0]})
    mask = pd.DataFrame()

    result = cli._apply_universe_mask(df, mask, date_column="date")

    assert result is df
