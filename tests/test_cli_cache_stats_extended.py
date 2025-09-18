from __future__ import annotations

import numpy as np
import pandas as pd

from trend_analysis import cli


def test_extract_cache_stats_prefers_last_snapshot() -> None:
    payload = {
        "snapshots": [
            {"entries": 1, "hits": 2, "misses": 3, "incremental_updates": 4},
            {
                "nested": [
                    {
                        "entries": 5.0,
                        "hits": 6.0,
                        "misses": 7.0,
                        "incremental_updates": 8.0,
                    }
                ],
                "frame": pd.DataFrame({"value": [1, 2, 3]}),
                "array": np.arange(3),
            },
        ]
    }

    stats = cli._extract_cache_stats(payload)

    assert stats == {
        "entries": 5,
        "hits": 6,
        "misses": 7,
        "incremental_updates": 8,
    }


def test_extract_cache_stats_returns_none_when_missing() -> None:
    payload = {
        "stats": {
            "entries": "bad",
            "hits": 2,
            "misses": 3,
            "incremental_updates": 4,
        },
        "sequence": [pd.Series([1, 2, 3], name="skip")],
    }

    assert cli._extract_cache_stats(payload) is None
