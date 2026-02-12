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


def test_load_nav_paths_reads_bundle_nav_paths_parquet_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    expected_parquet = bundle_dir / "nav_paths.parquet"
    expected_parquet.write_text("placeholder", encoding="utf-8")
    (bundle_dir / "nav_paths.csv").write_text("placeholder", encoding="utf-8")
    nested_dir = bundle_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "nav_paths.parquet").write_text("placeholder", encoding="utf-8")

    observed_paths: list[Path] = []
    expected_frame = pd.DataFrame({"path_0": [1.0, 1.1]})

    def _fake_read_parquet(path: Path) -> pd.DataFrame:
        observed_paths.append(path)
        return expected_frame

    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)

    loaded = load_nav_paths(bundle_dir)

    assert observed_paths == [expected_parquet]
    assert loaded is expected_frame


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
