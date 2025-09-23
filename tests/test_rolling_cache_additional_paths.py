"""Additional coverage for ``trend_analysis.perf.rolling_cache`` helpers."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

import trend_analysis.perf.rolling_cache as rolling_cache


def _reload_cache_module() -> None:
    importlib.reload(rolling_cache)


def test_default_cache_dir_rejects_paths_outside_home(monkeypatch, tmp_path: Path) -> None:
    original_home = Path.home
    monkeypatch.setenv("TREND_ROLLING_CACHE", "/var/tmp/trend-cache")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    _reload_cache_module()
    try:
        expected = tmp_path / ".cache/trend_model/rolling"
        assert rolling_cache._DEFAULT_CACHE_DIR == expected
    finally:
        monkeypatch.delenv("TREND_ROLLING_CACHE", raising=False)
        monkeypatch.setattr(Path, "home", original_home, raising=False)
        _reload_cache_module()


def test_normalise_component_replaces_invalid_characters() -> None:
    assert rolling_cache._normalise_component("risk@metric#1") == "risk_metric_1"


def test_compute_dataset_hash_rejects_unknown_types() -> None:
    with pytest.raises(TypeError):
        rolling_cache.compute_dataset_hash(["not-a-series"])  # type: ignore[list-item]


def test_get_or_compute_validates_return_type(tmp_path: Path) -> None:
    cache = rolling_cache.RollingCache(cache_dir=tmp_path)
    dataset_hash = "abc123"

    def compute():
        return pd.DataFrame({"a": [1]})

    with pytest.raises(TypeError):
        cache.get_or_compute(dataset_hash, 5, "M", "mean", compute)


def test_set_cache_enabled_toggles_global(monkeypatch, tmp_path: Path) -> None:
    cache = rolling_cache.RollingCache(cache_dir=tmp_path)
    monkeypatch.setattr(rolling_cache, "_DEFAULT_ROLLING_CACHE", cache)

    rolling_cache.set_cache_enabled(False)
    assert not cache.is_enabled()
    rolling_cache.set_cache_enabled(True)
    assert cache.is_enabled()
