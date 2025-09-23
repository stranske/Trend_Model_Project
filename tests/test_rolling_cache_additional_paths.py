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
    expected = tmp_path / ".cache/trend_model/rolling"
    assert rolling_cache._DEFAULT_CACHE_DIR == expected


def test_default_cache_dir_accepts_home_relative_paths(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "custom-cache"
    monkeypatch.setenv("TREND_ROLLING_CACHE", str(target))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    _reload_cache_module()

    assert rolling_cache._DEFAULT_CACHE_DIR == target.resolve()


def test_default_cache_dir_recovers_when_home_resolution_fails(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "allowed"
    monkeypatch.setenv("TREND_ROLLING_CACHE", str(target))
    call_count = {"value": 0}

    def flaky_home() -> Path:
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("boom")
        return tmp_path

    monkeypatch.setattr(Path, "home", flaky_home)

    _reload_cache_module()

    expected = tmp_path / ".cache/trend_model/rolling"
    assert rolling_cache._DEFAULT_CACHE_DIR == expected


def test_default_cache_dir_returns_standard_location_when_unset(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TREND_ROLLING_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    _reload_cache_module()

    expected = tmp_path / ".cache/trend_model/rolling"
    assert rolling_cache._DEFAULT_CACHE_DIR == expected
def test_normalise_component_replaces_invalid_characters() -> None:
    assert rolling_cache._normalise_component("risk@metric#1") == "risk_metric_1"


def test_compute_dataset_hash_rejects_unknown_types() -> None:
    with pytest.raises(TypeError):
        rolling_cache.compute_dataset_hash(["not-a-series"])  # type: ignore[list-item]


def test_compute_dataset_hash_is_deterministic_for_series_and_frames() -> None:
    series = pd.Series([1.0, 2.0], name="alpha")
    frame = pd.DataFrame({"beta": [3.0, 4.0]})

    first = rolling_cache.compute_dataset_hash([series, frame])
    second = rolling_cache.compute_dataset_hash([series, frame])

    assert first == second
    assert len(first) == 64


def test_get_or_compute_validates_return_type(tmp_path: Path) -> None:
    cache = rolling_cache.RollingCache(cache_dir=tmp_path)
    dataset_hash = "abc123"

    def compute():
        return pd.DataFrame({"a": [1]})

    with pytest.raises(TypeError):
        cache.get_or_compute(dataset_hash, 5, "M", "mean", compute)


def test_get_or_compute_honours_disabled_cache(tmp_path: Path) -> None:
    cache = rolling_cache.RollingCache(cache_dir=tmp_path)
    cache.set_enabled(False)
    dataset_hash = "abc123"

    calls: list[int] = []

    def compute() -> pd.Series:
        calls.append(1)
        return pd.Series([1.0], name="value")

    result = cache.get_or_compute(dataset_hash, 3, "M", "mean", compute)

    assert isinstance(result, pd.Series)
    assert calls == [1]
    assert not any(tmp_path.iterdir())


def test_get_or_compute_reuses_serialised_results(tmp_path: Path) -> None:
    cache = rolling_cache.RollingCache(cache_dir=tmp_path)
    dataset_hash = "abc123"

    calls: list[int] = []

    def compute() -> pd.Series:
        calls.append(1)
        return pd.Series([1.0], name="value")

    first = cache.get_or_compute(dataset_hash, 12, "M", "ema", compute)
    second = cache.get_or_compute(dataset_hash, 12, "M", "ema", compute)

    assert list(first) == [1.0]
    assert list(second) == [1.0]
    assert calls == [1]


def test_set_cache_enabled_toggles_global(monkeypatch, tmp_path: Path) -> None:
    cache = rolling_cache.RollingCache(cache_dir=tmp_path)
    monkeypatch.setattr(rolling_cache, "_DEFAULT_ROLLING_CACHE", cache)

    rolling_cache.set_cache_enabled(False)
    assert not cache.is_enabled()
    rolling_cache.set_cache_enabled(True)
    assert cache.is_enabled()


def test_get_cache_returns_default_singleton(monkeypatch, tmp_path: Path) -> None:
    cache = rolling_cache.RollingCache(cache_dir=tmp_path)
    monkeypatch.setattr(rolling_cache, "_DEFAULT_ROLLING_CACHE", cache)

    assert rolling_cache.get_cache() is cache
