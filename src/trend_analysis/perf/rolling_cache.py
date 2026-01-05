"""Persistent cache for expensive rolling computations."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import pandas as pd
from pandas.util import hash_pandas_object

from trend_analysis.util.joblib_shim import dump, load

from .timing import log_timing


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser()


def _get_home_dir() -> Path:
    try:
        return _safe_resolve(Path.home())
    except Exception:
        return _safe_resolve(Path(tempfile.gettempdir()))


def _get_default_cache_dir() -> Path:
    env_path = os.getenv("TREND_ROLLING_CACHE")
    home = _get_home_dir()
    if env_path:
        cache_path = _safe_resolve(Path(env_path))
        try:
            if not cache_path.is_relative_to(home):
                cache_path = home / ".cache/trend_model/rolling"
        except ValueError:
            cache_path = home / ".cache/trend_model/rolling"
        return cache_path
    return home / ".cache/trend_model/rolling"


_DEFAULT_CACHE_DIR = _get_default_cache_dir()


def _normalise_component(component: str) -> str:
    """Return a filesystem-safe version of ``component``."""

    return re.sub(r"[^A-Za-z0-9_.-]", "_", component)


def compute_dataset_hash(objects: Sequence[pd.Series | pd.DataFrame]) -> str:
    """Return a stable SHA-256 hash for the provided pandas objects."""

    hasher = hashlib.sha256()

    def _update_from_series(series: pd.Series) -> None:
        hashed = hash_pandas_object(series, index=True)
        hasher.update(hashed.to_numpy().tobytes())
        name = "" if series.name is None else str(series.name)
        hasher.update(name.encode("utf-8", "ignore"))
        hasher.update(str(series.dtype).encode("utf-8", "ignore"))

    for obj in objects:
        if isinstance(obj, pd.Series):
            _update_from_series(obj)
            continue
        if isinstance(obj, pd.DataFrame):
            for column in obj.columns:
                _update_from_series(obj[column])
            continue
        raise TypeError("compute_dataset_hash accepts Series or DataFrame instances")

    return hasher.hexdigest()


class RollingCache:
    """Filesystem-backed cache for rolling computations."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = (cache_dir or _DEFAULT_CACHE_DIR).expanduser()
        self._enabled = True
        self.cache_dir = self._ensure_cache_dir(self.cache_dir)

    def _ensure_cache_dir(self, cache_dir: Path) -> Path:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
        except (OSError, PermissionError):
            fallback = (Path(tempfile.gettempdir()) / "trend_model" / "rolling").resolve()
            try:
                fallback.mkdir(parents=True, exist_ok=True)
                return fallback
            except (OSError, PermissionError):
                self._enabled = False
                return cache_dir

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        return self._enabled

    def _build_path(self, dataset_hash: str, window: int, freq: str, method: str) -> Path:
        safe_method = _normalise_component(method)
        safe_freq = _normalise_component(freq)
        file_name = f"{dataset_hash}_{safe_method}_{safe_freq}_{window}.joblib"
        return self.cache_dir / file_name

    def get_or_compute(
        self,
        dataset_hash: str,
        window: int,
        freq: str,
        method: str,
        compute_fn: Callable[[], pd.Series],
    ) -> pd.Series:
        """Return cached result or compute and persist the series."""

        start = perf_counter()

        if not self._enabled:
            result = compute_fn()
            log_timing(
                "rolling_cache",
                duration_s=perf_counter() - start,
                status="disabled",
                method=method,
                window=window,
                freq=freq,
                hash=dataset_hash[:12],
            )
            return result

        cache_path = self._build_path(dataset_hash, window, freq, method)
        if cache_path.exists():
            try:
                cached = load(cache_path)
                if isinstance(cached, pd.Series):
                    log_timing(
                        "rolling_cache",
                        duration_s=perf_counter() - start,
                        status="hit",
                        method=method,
                        window=window,
                        freq=freq,
                        hash=dataset_hash[:12],
                    )
                    return cached
            except Exception:  # pragma: no cover - cache corruption fallback
                cache_path.unlink(missing_ok=True)

        result = compute_fn()
        if not isinstance(result, pd.Series):  # pragma: no cover - defensive
            raise TypeError("compute_fn must return a pandas Series")
        dump(result, cache_path)
        log_timing(
            "rolling_cache",
            duration_s=perf_counter() - start,
            status="miss",
            method=method,
            window=window,
            freq=freq,
            hash=dataset_hash[:12],
        )
        return result


_DEFAULT_ROLLING_CACHE = RollingCache()


def get_cache() -> RollingCache:
    """Return the process-wide rolling cache."""

    return _DEFAULT_ROLLING_CACHE


def set_cache_enabled(enabled: bool) -> None:
    """Globally enable or disable rolling cache usage."""

    _DEFAULT_ROLLING_CACHE.set_enabled(enabled)


__all__ = [
    "RollingCache",
    "compute_dataset_hash",
    "get_cache",
    "set_cache_enabled",
]
