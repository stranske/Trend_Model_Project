"""Lightweight performance caching utilities.

Provides a covariance (and related aggregates) cache to avoid repeated
O(N^2) computations across overlapping multi-period windows.

Design goals:
 - Pure in-memory dictionary; no eviction (Phase 1) but LRU-ready.
 - Deterministic keys based on (start, end, universe_hash, freq_tag).
 - Payload keeps raw aggregates to enable future incremental updates.
 - Zero external dependencies beyond NumPy / pandas.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .._typing import FloatArray, MatrixF
from .timing import log_timing

Key = Tuple[str, str, int, str]


def _universe_hash(assets: Iterable[str]) -> int:
    """Stable hash for an asset universe.

    Uses SHA-256 for stability across processes; truncated to 8 bytes ->
    int.
    """
    joined = "\x1f".join(sorted(map(str, assets)))
    h = hashlib.sha256(joined.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, "big", signed=False)


@dataclass(slots=True)
class CovPayload:
    cov: MatrixF
    mean: FloatArray
    std: FloatArray
    n: int
    assets: tuple[str, ...]
    # Optional aggregates enabling O(k^2) rolling updates (None unless requested)
    s1: FloatArray | None = None  # sum of rows (vector)
    s2: MatrixF | None = None  # sum of outer products (matrix)

    def as_dict(self) -> Dict[str, Any]:  # convenience for tests / debug
        return {
            "cov": self.cov,
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
            "assets": self.assets,
        }


class CovCache:
    """In-memory covariance cache with simple (optional) LRU behaviour.

    Parameters
    ----------
    capacity : int | None
        Max entries to retain (oldest evicted). ``None`` = unlimited.
    lru : bool
        Whether to treat the store as an OrderedDict implementing LRU touches.
    """

    def __init__(self, capacity: int | None = None, *, lru: bool = True) -> None:
        self.capacity = capacity
        self.lru = lru
        self._store: "OrderedDict[Key, CovPayload]" = OrderedDict()
        # stats
        self.hits = 0
        self.misses = 0
        self.incremental_updates = 0  # external code increments when used

    # -- core API -----------------------------------------------------
    def get(self, key: Key) -> CovPayload | None:
        payload = self._store.get(key)
        if payload is not None and self.lru:
            # move to end to mark as recently used
            self._store.move_to_end(key)
            self.hits += 1
        elif payload is None:
            self.misses += 1
        return payload

    def put(self, key: Key, payload: CovPayload) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = payload
        if self.capacity is not None and len(self._store) > self.capacity:
            # Evict oldest
            self._store.popitem(last=False)

    def get_or_compute(
        self, key: Key, compute_fn: Callable[[], CovPayload]
    ) -> CovPayload:
        start = perf_counter()
        cached = self.get(key)
        if cached is not None:
            log_timing(
                "cov_cache",
                duration_s=perf_counter() - start,
                status="hit",
                start=key[0],
                end=key[1],
                freq=key[3],
                assets=len(cached.assets),
            )
            return cached
        payload = compute_fn()
        self.put(key, payload)
        log_timing(
            "cov_cache",
            duration_s=perf_counter() - start,
            status="miss",
            start=key[0],
            end=key[1],
            freq=key[3],
            assets=len(payload.assets),
        )
        return payload

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = self.incremental_updates = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)

    # -- convenience --------------------------------------------------
    @staticmethod
    def make_key(start: str, end: str, assets: Iterable[str], freq: str = "M") -> Key:
        return (start, end, _universe_hash(assets), freq)

    # -- statistics --------------------------------------------------
    def stats(self) -> dict[str, int]:  # pragma: no cover - trivial
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "incremental_updates": self.incremental_updates,
        }


def compute_cov_payload(
    df: pd.DataFrame, *, materialise_aggregates: bool = False
) -> CovPayload:
    """Compute covariance + aggregates for the provided returns frame.

    Parameters
    ----------
    df : pd.DataFrame
        Return series (rows=time, cols=assets).
    materialise_aggregates : bool, default False
        When True, store S1 (sum of rows) and S2 (sum of outer products) to
        enable future incremental (rolling) updates without full recompute.

    Notes
    -----
    Frame must contain only asset return columns (no Date). NaNs are
    forward-filled then remaining missing replaced with zeros (conservative).
    """
    if df.empty:
        raise ValueError("DataFrame is empty for covariance computation")
    assets = tuple(map(str, df.columns))
    arr = df.astype(float).copy()
    # Forward fill then remaining gaps -> 0.0 (conservative). Avoid deprecated fillna(method='ffill').
    arr = arr.ffill().fillna(0.0)
    values = arr.to_numpy(dtype=float)
    n = values.shape[0]
    if n < 2:
        # Degenerate: zero covariance, std zeros
        cov = np.zeros((len(assets), len(assets)), dtype=float)
        mean = values.mean(axis=0)
        std = np.zeros(len(assets), dtype=float)
        return CovPayload(cov=cov, mean=mean, std=std, n=n, assets=assets)
    mean = values.mean(axis=0)
    demeaned = values - mean
    cov = (demeaned.T @ demeaned) / (n - 1)
    std = np.sqrt(np.diag(cov))
    if materialise_aggregates:
        s1 = values.sum(axis=0)
        s2 = values.T @ values  # X^T X with shape (k,k)
    else:
        s1 = None
        s2 = None
    return CovPayload(cov=cov, mean=mean, std=std, n=n, assets=assets, s1=s1, s2=s2)


def _ensure_aggregates(payload: CovPayload) -> None:
    """Populate S1/S2 if absent (reconstruct from existing stats).

    Uses identity: cov = (S2 - n * m m^T) / (n-1)  ->  S2 = cov*(n-1) + n * m m^T
    S1 is simply m * n.
    """
    if payload.s1 is not None and payload.s2 is not None:
        return
    n = payload.n
    if n < 1:
        raise ValueError("Cannot reconstruct aggregates with n < 1")
    mean = payload.mean
    cov = payload.cov
    outer_mm = np.outer(mean, mean)
    s2 = cov * (n - 1) + n * outer_mm
    s1 = mean * n
    payload.s1 = s1
    payload.s2 = s2


def incremental_cov_update(
    prev: CovPayload,
    old_row: FloatArray,
    new_row: FloatArray,
) -> CovPayload:
    """Incrementally update covariance for a sliding window.

    Parameters
    ----------
    prev : CovPayload
        Previous window payload.
    old_row : np.ndarray
        Vector leaving the window (asset order must match payload.assets).
    new_row : np.ndarray
        Vector entering the window.

    Returns
    -------
    CovPayload
        Updated payload (shares asset ordering).
    """
    if old_row.shape != new_row.shape:
        raise ValueError("old_row and new_row must have identical shape")
    if old_row.shape[0] != len(prev.assets):
        raise ValueError("Row length does not match number of assets")
    n = prev.n
    if n < 2:
        raise ValueError("Incremental update requires n >= 2")

    _ensure_aggregates(prev)
    assert prev.s1 is not None and prev.s2 is not None  # for mypy / type checkers

    # Update aggregates: S1' = S1 - old + new ; S2' = S2 - oo^T + nn^T
    s1_new = prev.s1 - old_row + new_row
    s2_new = prev.s2 - np.outer(old_row, old_row) + np.outer(new_row, new_row)
    mean_new = s1_new / n
    cov_new = (s2_new - n * np.outer(mean_new, mean_new)) / (n - 1)
    std_new = np.sqrt(np.diag(cov_new))
    return CovPayload(
        cov=cov_new,
        mean=mean_new,
        std=std_new,
        n=n,
        assets=prev.assets,
        s1=s1_new,
        s2=s2_new,
    )


__all__ = [
    "CovCache",
    "CovPayload",
    "compute_cov_payload",
    "incremental_cov_update",
]
