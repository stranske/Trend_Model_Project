from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend.mc.charts import NAV_PATH_REQUIRED_CHARTS
from trend.mc.io import (
    MCNavPathsIOError,
    MISSING_NAV_PATHS_RAISE,
    MISSING_NAV_PATHS_RETURN_NONE,
    load_nav_paths,
    validate_nav_paths_df,
    validate_nav_paths_requirement,
)


def test_load_nav_paths_returns_none_when_parquet_missing_and_optional(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    assert (
        load_nav_paths(bundle_dir, missing_parquet=MISSING_NAV_PATHS_RETURN_NONE) is None
    )


def test_load_nav_paths_raises_when_parquet_missing_and_required(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    with pytest.raises(MCNavPathsIOError, match=r"Missing required nav_paths\.parquet file"):
        load_nav_paths(bundle_dir, missing_parquet=MISSING_NAV_PATHS_RAISE)


def test_validate_nav_paths_df_rejects_non_dataframe() -> None:
    with pytest.raises(MCNavPathsIOError, match="must be a pandas DataFrame"):
        validate_nav_paths_df({"nav": [1, 2]})


def test_validate_nav_paths_df_rejects_empty_dataframe() -> None:
    with pytest.raises(MCNavPathsIOError, match="must not be empty"):
        validate_nav_paths_df(pd.DataFrame())


def test_validate_nav_paths_df_rejects_missing_required_columns() -> None:
    frame = pd.DataFrame({"date": ["2026-01-01"], "nav": [100.0]})

    with pytest.raises(MCNavPathsIOError, match=r"missing required column\(s\): path_id"):
        validate_nav_paths_df(frame, required_columns=("date", "nav", "path_id"))


def test_validate_nav_paths_requirement_uses_nav_path_required_charts_constant() -> None:
    assert NAV_PATH_REQUIRED_CHARTS == frozenset({"path_dist"})
    with pytest.raises(MCNavPathsIOError, match=r"path_dist require nav_paths\.parquet"):
        validate_nav_paths_requirement(
            selected_charts=["fan", "path_dist"],
            nav_paths_frame=None,
            nav_path_required_charts=NAV_PATH_REQUIRED_CHARTS,
        )
