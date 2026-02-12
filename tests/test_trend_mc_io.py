from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend.mc.io import MCNavPathsIOError, load_nav_paths_frame, validate_nav_paths_requirement


def test_load_nav_paths_frame_returns_none_when_optional_file_missing(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    assert load_nav_paths_frame(bundle_dir) is None


@pytest.mark.parametrize("suffix", ("csv", "json"))
def test_load_nav_paths_frame_errors_for_unsupported_formats_without_parquet(
    tmp_path: Path, suffix: str
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / f"nav_paths.{suffix}").write_text("placeholder", encoding="utf-8")

    with pytest.raises(MCNavPathsIOError, match=r"Only nav_paths\.parquet is supported"):
        load_nav_paths_frame(bundle_dir)


def test_load_nav_paths_frame_reads_parquet_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    parquet_path = bundle_dir / "nav_paths.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    expected = pd.DataFrame({"path": [1, 2, 3]})
    monkeypatch.setattr(pd, "read_parquet", lambda _path: expected)

    loaded = load_nav_paths_frame(bundle_dir)

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, expected)


def test_validate_nav_paths_requirement_raises_when_required_chart_requested_without_nav_paths() -> None:
    with pytest.raises(MCNavPathsIOError, match=r"require nav_paths\.parquet"):
        validate_nav_paths_requirement(
            ["fan", "path_dist"],
            None,
            nav_path_required_charts=frozenset({"path_dist"}),
        )


def test_validate_nav_paths_requirement_allows_non_required_charts_without_nav_paths() -> None:
    validate_nav_paths_requirement(
        ["fan", "risk_return"],
        None,
        nav_path_required_charts=frozenset({"path_dist"}),
    )
