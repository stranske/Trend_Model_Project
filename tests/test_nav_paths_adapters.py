from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.viz.adapters import PATHS_INDEX_NAMES, make_paths


def test_nav_paths_adapter_filters_to_nav_asset_level() -> None:
    index = pd.date_range("2024-01-31", periods=2, freq="ME")
    columns = pd.MultiIndex.from_tuples(
        [("path_a", "NAV"), ("path_a", "PX_LAST"), ("path_b", "NAV")],
        names=["path", "asset"],
    )
    nav_paths = pd.DataFrame(
        [
            [1.00, 101.0, 1.00],
            [1.05, 105.0, 1.02],
        ],
        index=index,
        columns=columns,
    )

    canonical = make_paths(nav_paths)

    assert tuple(canonical.index.names) == PATHS_INDEX_NAMES
    assert canonical.shape == (4, 1)
    assert canonical.loc[(pd.Timestamp("2024-02-29"), "path_a"), "nav"] == pytest.approx(1.05)
    assert canonical.loc[(pd.Timestamp("2024-02-29"), "path_b"), "nav"] == pytest.approx(1.02)


def test_nav_paths_adapter_collapses_duplicate_paths_by_mean() -> None:
    nav_paths = pd.DataFrame(
        {
            "path_1": [1.0, 1.1],
            "path_1_dup": [1.2, 1.5],
            "path_2": [0.9, 1.0],
        },
        index=pd.date_range("2024-01-31", periods=2, freq="ME"),
    )
    nav_paths.columns = pd.Index(["path_1", "path_1", "path_2"], name="path")

    canonical = make_paths(nav_paths)

    assert canonical.loc[(pd.Timestamp("2024-01-31"), "path_1"), "nav"] == pytest.approx(1.1)
    assert canonical.loc[(pd.Timestamp("2024-02-29"), "path_1"), "nav"] == pytest.approx(1.3)
    assert canonical.loc[(pd.Timestamp("2024-02-29"), "path_2"), "nav"] == pytest.approx(1.0)


def test_nav_paths_adapter_empty_input_returns_empty_canonical_frame() -> None:
    nav_paths = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    canonical = make_paths(nav_paths)

    assert canonical.empty
    assert tuple(canonical.index.names) == PATHS_INDEX_NAMES
    assert list(canonical.columns) == ["nav"]
    assert canonical["nav"].dtype == "float64"
