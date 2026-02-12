from __future__ import annotations

import pandas as pd
import pytest

from trend_analysis.viz.adapters import (
    PATHS_INDEX_NAMES,
    PATHS_REQUIRED_COLUMNS,
    PATHS_REQUIRED_DTYPES,
    make_paths,
)


def _sample_nav_paths() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path_1": [1.0, 1.02, 1.05],
            "path_2": [1.0, 0.98, 1.01],
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )


def test_nav_paths_contract_make_paths_emits_canonical_schema() -> None:
    canonical = make_paths(_sample_nav_paths())

    assert isinstance(canonical.index, pd.MultiIndex)
    assert tuple(canonical.index.names) == PATHS_INDEX_NAMES
    assert tuple(canonical.columns) == PATHS_REQUIRED_COLUMNS
    assert canonical["nav"].dtype == PATHS_REQUIRED_DTYPES["nav"]


def test_nav_paths_contract_requires_datetime_like_index() -> None:
    nav_paths = pd.DataFrame(
        {"path_1": [1.0, 1.01], "path_2": [1.0, 0.99]},
        index=["not-a-date", "also-not-a-date"],
    )

    with pytest.raises(ValueError, match="datetime-like"):
        make_paths(nav_paths)


def test_nav_paths_contract_rejects_non_dataframe_input() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        make_paths({"path_1": [1.0, 1.01]})  # type: ignore[arg-type]
