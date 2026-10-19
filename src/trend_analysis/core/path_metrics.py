"""Explicit, path-scoped metric reuse for single- and multi-period evaluation.

The context variable transports an optional cache through synchronous pipeline
stages without putting runtime state in persisted strategy configuration. Each
path owns its mapping; context tokens are always restored, including on errors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import pandas as pd

MetricCache = dict[tuple[object, ...], pd.Series]
_ACTIVE_CACHE: ContextVar[MetricCache | None] = ContextVar("path_metric_cache", default=None)


@contextmanager
def use_metric_cache(cache: MetricCache) -> Iterator[None]:
    token = _ACTIVE_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_CACHE.reset(token)


def _frame_key(value: pd.DataFrame | pd.Series) -> tuple[object, ...]:
    frame = value.to_frame() if isinstance(value, pd.Series) else value
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return (
        tuple(frame.columns),
        tuple(str(dtype) for dtype in frame.dtypes),
        str(frame.index.dtype),
        digest.hexdigest(),
    )


def compute_metric_series(
    frame: pd.DataFrame, metric: str, stats_cfg: Any, **kwargs: Any
) -> pd.Series:
    """Reuse only identical effective metric inputs, returning independent values."""
    from .rank_selection import METRIC_REGISTRY, _compute_metric_series

    cache = _ACTIVE_CACHE.get()
    if cache is None:
        return _compute_metric_series(frame, metric, stats_cfg, **kwargs)
    # These are the effective arguments consumed by the metric kernel. Registry
    # lists do not affect an individual metric: lazy population therefore builds
    # the union required by compatible strategies without recomputing overlaps.
    risk_free = kwargs.get("risk_free_override")
    if risk_free is None:
        risk_free = stats_cfg.risk_free
    benchmark = kwargs.get("benchmark")
    if benchmark is None:
        benchmark = stats_cfg.benchmark
    inputs = tuple(
        _frame_key(value) if isinstance(value, (pd.DataFrame, pd.Series)) else repr(value)
        for value in (risk_free, benchmark, stats_cfg.periods_per_year)
    )
    key = (_frame_key(frame), metric, id(METRIC_REGISTRY.get(metric)), inputs)
    if key not in cache:
        cache[key] = _compute_metric_series(frame, metric, stats_cfg, **kwargs).copy()
    return cache[key].copy()
