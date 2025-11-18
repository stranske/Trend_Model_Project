"""Persistent cache for expensive rolling computations."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import pandas as pd
from pandas.util import hash_pandas_object

from trend_analysis.util.joblib_shim import dump, load
from .timing import log_timing


def _get_default_cache_dir() -> Path:
    env_path = os.getenv("TREND_ROLLING_CACHE")
    if env_path:
        # Expand user and resolve to absolute path
        cache_path = Path(env_path).expanduser().resolve()
        # Optionally, ensure cache_path is within the user's home directory
        try:
            home = Path.home().resolve()
            if not str(cache_path).startswith(str(home)):
                # Fallback to safe default if outside home
                cache_path = home / ".cache/trend_model/rolling"
        except Exception:
            cache_path = Path.home() / ".cache/trend_model/rolling"
        return cache_path
    else:
        return Path.home() / ".cache/trend_model/rolling"


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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        return self._enabled

    def _build_path(
        self, dataset_hash: str, window: int, freq: str, method: str
    ) -> Path:
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
