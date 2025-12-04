from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_analysis.cli import (
    _apply_universe_mask,
    _extract_cache_stats,
    _resolve_artifact_paths,
)


def test_resolve_artifact_paths_deduplicates_and_normalises_excel_alias():
    resolved = _resolve_artifact_paths(
        Path("/tmp/out"),
        "result",
        ["summary", "details"],
        ["csv", "excel", "json", "CSV"],
    )
    assert resolved == [
        Path("/tmp/out/result_summary.csv"),
        Path("/tmp/out/result_details.csv"),
        Path("/tmp/out/result.xlsx"),
        Path("/tmp/out/result_summary.json"),
        Path("/tmp/out/result_details.json"),
    ]


def test_extract_cache_stats_returns_last_integer_like_snapshot():
    payload = {
        "meta": {"hits": 1.0, "entries": 2, "misses": 3, "incremental_updates": 0},
        "nested": [
            {"entries": 5, "hits": 4.2, "misses": 1, "incremental_updates": 3},
            np.array([1, 2, 3]),
            {
                "entries": 10,
                "hits": 9,
                "misses": 1,
                "incremental_updates": 7,
                "extra": "ignored",
            },
        ],
    }

    assert _extract_cache_stats(payload) == {
        "entries": 10,
        "hits": 9,
        "misses": 1,
        "incremental_updates": 7,
    }


def test_extract_cache_stats_ignores_invalid_candidates():
    class CustomSized:
        def __len__(self):
            return 3

    payload = [
        {"entries": "a", "hits": 1, "misses": 1, "incremental_updates": 1},
        pd.DataFrame({"a": [1, 2]}),
        CustomSized(),
    ]

    assert _extract_cache_stats(payload) is None


def test_apply_universe_mask_respects_membership_and_date_column_case_insensitive():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [4.0, 5.0, 6.0],
        }
    )
    mask = pd.DataFrame(
        {
            "AAA": [True, False, True],
            "BBB": [False, True, False],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    result = _apply_universe_mask(df, mask, date_column="date")

    expected = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "AAA": [1.0, np.nan, 3.0],
            "BBB": [np.nan, 5.0, np.nan],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_apply_universe_mask_raises_for_missing_member_columns():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "AAA": [1.0, 2.0],
        }
    )
    mask = pd.DataFrame(
        {
            "AAA": [True, True],
            "BBB": [False, True],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    with pytest.raises(KeyError) as excinfo:
        _apply_universe_mask(df, mask, date_column="date")

    assert "BBB" in str(excinfo.value)
