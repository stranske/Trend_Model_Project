"""Parquet-path integration gate for the MC bundle loaders.

Issue #5347: ``pyarrow`` is required to read the parquet MC bundles
(``summary.parquet``, ``results.parquet``, ``nav_paths.parquet``) shipped in
``tests/fixtures/mc_bundle`` and consumed by ``trend mc viz``. Before pyarrow
became an explicit ``[project].dependencies`` entry it was only present
transitively via Streamlit, so a base ``pip install -e .`` (no ``[app]``) could
not read these fixtures at all.

The existing ``tests/integration/test_mc_viz.py`` drives the CLI end-to-end;
this module adds an in-process gate that loads the parquet fixture directly
through the production loaders, so the parquet code path is exercised even when
the heavier CLI/PNG plumbing is unavailable. It fails in a base-deps-only
environment without pyarrow and passes once pyarrow is installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trend.mc.io import load_nav_paths_frame
from trend.mc.viz import _load_mc_bundle_frames

pytestmark = [pytest.mark.integration, pytest.mark.mc_viz_integration]


def _fixture_bundle_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "mc_bundle"


def test_pyarrow_is_importable() -> None:
    """pyarrow must be importable for the parquet bundle path to work."""
    import pyarrow

    assert pyarrow.__version__


def test_mc_bundle_summary_and_results_parquet_load() -> None:
    """summary.parquet and results.parquet load into non-empty DataFrames."""
    bundle_dir = _fixture_bundle_dir()
    assert (bundle_dir / "summary.parquet").is_file()
    assert (bundle_dir / "results.parquet").is_file()

    summary, results = _load_mc_bundle_frames(bundle_dir)

    assert isinstance(summary, pd.DataFrame)
    assert isinstance(results, pd.DataFrame)
    assert not summary.empty
    assert not results.empty


def test_mc_bundle_nav_paths_parquet_loads() -> None:
    """nav_paths.parquet loads into a non-empty DataFrame via the io loader."""
    bundle_dir = _fixture_bundle_dir()
    assert (bundle_dir / "nav_paths.parquet").is_file()

    nav_paths = load_nav_paths_frame(bundle_dir)

    assert isinstance(nav_paths, pd.DataFrame)
    assert not nav_paths.empty
