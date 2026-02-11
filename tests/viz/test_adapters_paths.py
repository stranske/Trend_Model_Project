from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_analysis.viz.adapters import (
    PATHS_INDEX_NAMES,
    PATHS_REQUIRED_COLUMNS,
    PATHS_REQUIRED_DTYPES,
    ROLLING_REQUIRED_COLUMNS,
    _normalize_nav_paths,
    make_paths,
    path_correlations,
    rolling_stats,
    terminal_returns,
)


def _sample_nav_paths() -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    return pd.DataFrame(
        {
            0: [1.0, 1.1, 1.21],
            1: [1.0, 0.9, 0.99],
        },
        index=index,
    )


def test_make_paths_includes_required_schema_and_index_structure() -> None:
    canonical = make_paths(_sample_nav_paths())

    assert isinstance(canonical.index, pd.MultiIndex)
    assert tuple(canonical.index.names) == PATHS_INDEX_NAMES
    for col in PATHS_REQUIRED_COLUMNS:
        assert col in canonical.columns
    assert canonical.dtypes["nav"] == PATHS_REQUIRED_DTYPES["nav"]
    assert canonical.shape == (6, 1)


def test_make_paths_supports_multiindex_columns_with_asset_level() -> None:
    base = _sample_nav_paths()
    columns = pd.MultiIndex.from_tuples([(0, "NAV"), (1, "NAV")], names=["path", "asset"])
    nav_paths = base.copy()
    nav_paths.columns = columns

    canonical = make_paths(nav_paths)

    assert canonical.shape == (6, 1)
    assert canonical.loc[(pd.Timestamp("2024-03-31"), 0), "nav"] == 1.21


def test_normalize_nav_paths_collapses_duplicate_path_labels_by_mean() -> None:
    index = pd.to_datetime(["2024-01-31", "2024-02-29"])
    nav_paths = pd.DataFrame(
        np.array(
            [
                [1.0, 1.2, 0.9],
                [1.1, 1.5, 1.0],
            ]
        ),
        index=index,
        columns=pd.Index([0, 0, 1], name="path"),
    )

    normalized = _normalize_nav_paths(nav_paths)

    assert list(normalized.columns) == [0, 1]
    assert normalized.loc[pd.Timestamp("2024-01-31"), 0] == pytest.approx(1.1)
    assert normalized.loc[pd.Timestamp("2024-02-29"), 0] == pytest.approx(1.3)
    assert normalized.loc[pd.Timestamp("2024-02-29"), 1] == pytest.approx(1.0)


def test_make_paths_rejects_multiindex_asset_without_nav() -> None:
    base = _sample_nav_paths()
    columns = pd.MultiIndex.from_tuples([(0, "PX_LAST"), (1, "PX_LAST")], names=["path", "asset"])
    nav_paths = base.copy()
    nav_paths.columns = columns

    with pytest.raises(ValueError, match="must include 'NAV'"):
        make_paths(nav_paths)


def test_terminal_returns_full_horizon_and_lookback() -> None:
    canonical = make_paths(_sample_nav_paths())

    full = terminal_returns(canonical)
    lookback = terminal_returns(canonical, lookback_periods=1)

    assert full.loc[0, "terminal_return"] == pytest.approx(0.21)
    assert full.loc[1, "terminal_return"] == pytest.approx(-0.01)
    assert full.loc[0, "lookback_periods"] == 2
    assert lookback.loc[0, "terminal_return"] == pytest.approx(0.1)
    assert lookback.loc[1, "terminal_return"] == pytest.approx(0.1)


def test_rolling_stats_returns_expected_columns_and_values() -> None:
    canonical = make_paths(_sample_nav_paths())
    rolling = rolling_stats(canonical, window=2, periods_per_year=12)

    assert tuple(rolling.columns) == ROLLING_REQUIRED_COLUMNS
    assert tuple(rolling.index.names) == PATHS_INDEX_NAMES
    value = rolling.loc[(pd.Timestamp("2024-03-31"), 0), "rolling_mean"]
    assert value == pytest.approx(0.1)
    std_value = rolling.loc[(pd.Timestamp("2024-03-31"), 1), "rolling_std"]
    assert std_value == pytest.approx(0.1)


def test_path_correlations_returns_symmetric_matrix_with_unit_diagonal() -> None:
    canonical = make_paths(_sample_nav_paths())
    corr = path_correlations(canonical)

    assert list(corr.index) == [0, 1]
    assert list(corr.columns) == [0, 1]
    assert np.allclose(np.diag(corr.values), 1.0, equal_nan=False)
    assert corr.loc[0, 1] == corr.loc[1, 0]


def test_helper_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="nav_paths must be a pandas DataFrame"):
        make_paths({"a": [1.0]})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="window must be > 1"):
        rolling_stats(pd.DataFrame({"nav": []}), window=1)

    with pytest.raises(ValueError, match="window must be > 1"):
        path_correlations(pd.DataFrame({"nav": []}), window=1)
