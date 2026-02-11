from __future__ import annotations

import pandas as pd

from trend_analysis.viz.adapters import PATHS_INDEX_NAMES, make_paths


def _sample_nav_paths() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    return pd.DataFrame(
        {
            0: [1.0, 1.1, 1.21],
            1: [1.0, 0.9, 0.99],
        },
        index=index,
    )


def test_make_paths_required_schema_index_and_expected_shape() -> None:
    canonical = make_paths(_sample_nav_paths())

    assert isinstance(canonical.index, pd.MultiIndex)
    assert tuple(canonical.index.names) == PATHS_INDEX_NAMES
    assert tuple(canonical.columns) == ("nav",)
    assert canonical["nav"].dtype == "float64"
    assert canonical.shape == (6, 1)
