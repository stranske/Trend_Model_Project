from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "MetricCacheEntry",
    "MetricCache",
    "global_metric_cache",
    "make_metric_key",
    "get_or_compute_metric_series",
    "clear_metric_cache",
]


@dataclass(frozen=True)
class MetricCacheEntry:
    series: pd.Series
    # Potential future fields: timestamp, compute_cost_estimate


class MetricCache:
    def __init__(self) -> None:
        self._store: dict[str, MetricCacheEntry] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> pd.Series | None:
        entry = self._store.get(key)
        if entry is not None:
            self.hits += 1
            return entry.series
        self.misses += 1
        return None

    def put(self, key: str, series: pd.Series) -> pd.Series:
        # Store without copying; contract: callers do not mutate returned Series in-place
        self._store[key] = MetricCacheEntry(series=series)
        return series

    def clear(self) -> None:  # pragma: no cover - trivial
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


global_metric_cache = MetricCache()


def _hash_parts(parts: tuple[str, ...]) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def make_metric_key(
    start: str,
    end: str,
    universe_cols: tuple[str, ...],
    metric_name: str,
    cfg_hash: str | None = None,
) -> str:
    """Return deterministic cache key for scalar metric series.

    universe order matters.
    """
    return _hash_parts((start, end, ",".join(universe_cols), metric_name, cfg_hash or "_"))


def get_or_compute_metric_series(
    *,
    start: str,
    end: str,
    universe_cols: tuple[str, ...],
    metric_name: str,
    cfg_hash: str | None,
    compute: Callable[[], pd.Series],
    enable: bool,
    cache: MetricCache | None = None,
) -> pd.Series:
    """Centralised memoization helper for scalar metric series.

    If ``enable`` is False, bypass cache entirely.
    ``compute`` must return a Series indexed by ``universe_cols`` in the same order.
    """
    if not enable:
        return compute()
    mc = cache or global_metric_cache
    key = make_metric_key(start, end, universe_cols, metric_name, cfg_hash)
    cached = mc.get(key)
    if cached is not None:
        return cached
    series = compute()
    # Safety: ensure index alignment before storing
    if tuple(series.index.astype(str)) != universe_cols:
        # Reindex (will raise if missing) to guarantee consistency
        series = series.reindex(list(universe_cols))
    return mc.put(key, series)


def clear_metric_cache() -> None:  # pragma: no cover - trivial passthrough
    global_metric_cache.clear()
