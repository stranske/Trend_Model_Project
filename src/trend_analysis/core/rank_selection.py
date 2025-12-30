"""Rank-based fund selection utilities.

This module implements the `rank` selection mode described in Agents.md.
Funds can be kept using `top_n`, `top_pct` or `threshold` rules and scored
by metrics registered in `METRIC_REGISTRY`. Metrics listed in
`ASCENDING_METRICS` are treated as smaller-is-better.
"""

# =============================================================================
#  Runtime imports and dataclasses
# =============================================================================
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .. import metrics as _metrics

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..perf.cache import CovCache, CovPayload

# Compiled regex pattern for better performance when processing large files
_FIRM_NAME_TOKENIZER = re.compile(r"[^A-Za-z]+")

DEFAULT_METRIC = "annual_return"

WindowKey = tuple[str, str, str, str]


@dataclass
class RankSelectionDiagnostics:
    """Diagnostics for empty or failed rank selections."""

    reason: str
    metric: str
    transform: str
    inclusion_approach: str
    total_candidates: int
    non_null_scores: int
    threshold: float | None = None

    def message(self) -> str:
        details: list[str] = [
            f"metric={self.metric}",
            f"transform={self.transform}",
            f"approach={self.inclusion_approach}",
            f"total_candidates={self.total_candidates}",
            f"non_null_scores={self.non_null_scores}",
        ]
        if self.threshold is not None:
            details.append(f"threshold={self.threshold}")
        return f"{self.reason} ({', '.join(details)})"


def _json_default(value: Any) -> Any:
    """Helper for JSON serialisation of stats configuration objects."""

    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


def _canonicalise_labels(labels: Iterable[str]) -> list[str]:
    """Return column labels normalised for caching consistency."""

    clean: list[str] = []
    seen: set[str] = set()
    for idx, label in enumerate(labels):
        name = str(label).strip()
        if not name:
            name = f"Unnamed_{idx + 1}"
        base = name
        counter = 1
        while name in seen:
            counter += 1
            name = f"{base}_{counter}"
        seen.add(name)
        clean.append(name)
    return clean


def _ensure_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return *frame* with canonicalised column labels."""

    if frame.columns.empty:
        return frame
    clean = _canonicalise_labels(frame.columns)
    if list(frame.columns) == clean:
        return frame
    out = frame.copy()
    out.columns = clean
    return out


def _hash_universe(universe: Iterable[str]) -> str:
    joined = "\x1f".join(sorted(map(str, universe)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _stats_cfg_hash(cfg: RiskStatsConfig) -> str:
    base = asdict(cfg)
    extras = {k: v for k, v in vars(cfg).items() if k not in base}
    if extras:
        base["__extras__"] = extras
    payload = json.dumps(base, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class WindowMetricBundle:
    """Cached metric bundle for a selector window."""

    key: WindowKey | None
    start: str
    end: str
    freq: str
    stats_cfg_hash: str
    universe: tuple[str, ...]
    in_sample_df: pd.DataFrame
    _metrics: pd.DataFrame
    cov_payload: CovPayload | None = None

    def metrics_frame(self) -> pd.DataFrame:
        """Return a copy of the cached metrics frame."""

        if self._metrics.empty:
            return pd.DataFrame(index=self.in_sample_df.columns)
        return self._metrics.copy()

    def as_frame(self) -> pd.DataFrame:
        """Backward compatible alias for :meth:`metrics_frame`."""

        return self.metrics_frame()

    def available_metrics(self) -> list[str]:
        """Return the metric names cached in this bundle."""

        return list(self._metrics.columns)

    def ensure_metric(
        self,
        metric_name: str,
        stats_cfg: RiskStatsConfig,
        *,
        cov_cache: CovCache | None = None,
        enable_cov_cache: bool = True,
        incremental_cov: bool = False,
    ) -> pd.Series:
        """Ensure *metric_name* exists in the cached frame and return it."""

        canonical = _METRIC_ALIASES.get(metric_name, metric_name)
        if canonical in self._metrics.columns:
            _WINDOW_METRIC_CACHE.record_hit()
            _sync_cache_counters()
            return self._metrics[canonical]
        if canonical in {"AvgCorr", "__COV_VAR__"}:
            payload = self.cov_payload
            if payload is None:
                payload = _compute_covariance_payload(
                    self,
                    cov_cache,
                    enable_cov_cache=enable_cov_cache,
                    incremental_cov=incremental_cov,
                )
                self.cov_payload = payload
            series = _cov_metric_from_payload(canonical, payload, self.in_sample_df.columns)
        else:
            # Attempt scalar metric cache (Issue #1156)
            from .metric_cache import get_or_compute_metric_series as _metric_cached

            use_metric_cache = getattr(stats_cfg, "enable_metric_cache", False)
            series = _metric_cached(
                start=self.start,
                end=self.end,
                universe_cols=tuple(self.in_sample_df.columns.astype(str)),
                metric_name=canonical,
                cfg_hash=self.stats_cfg_hash,
                compute=lambda: _compute_metric_series(self.in_sample_df, canonical, stats_cfg),
                enable=use_metric_cache,
                cache=None,
            )
        series = series.astype(float)
        self._metrics[canonical] = series
        _WINDOW_METRIC_CACHE.record_miss()
        _sync_cache_counters()
        return self._metrics[canonical]


class WindowMetricCache:
    """Cache for selector window metric bundles with optional scoping."""

    def __init__(self, max_entries: int | None = None):
        self._max_entries = max_entries
        self._scoped: dict[str, OrderedDict[WindowKey, WindowMetricBundle]] = {}
        self._hits = 0
        self._misses = 0

    def _bucket(self) -> OrderedDict[WindowKey, WindowMetricBundle]:
        scope = _CACHE_SCOPE.get()
        return self._scoped.setdefault(scope, OrderedDict())

    def get(self, key: WindowKey) -> WindowMetricBundle | None:
        bucket = self._bucket()
        bundle = bucket.get(key)
        if bundle is not None:
            self._hits += 1
        return bundle

    def put(self, key: WindowKey | None, bundle: WindowMetricBundle) -> None:
        if key is None:
            return
        bucket = self._bucket()
        bucket[key] = bundle
        bucket.move_to_end(key)
        self._evict(bucket)

    def _evict(self, bucket: OrderedDict[WindowKey, WindowMetricBundle]) -> None:
        if self._max_entries is None:
            return
        while len(bucket) > self._max_entries:
            bucket.popitem(last=False)

    def set_limit(self, max_entries: int | None) -> int | None:
        previous = self._max_entries
        self._max_entries = max_entries
        if max_entries is None:
            return previous
        for bucket in self._scoped.values():
            self._evict(bucket)
        return previous

    def clear(self, scope: str | None = None) -> None:
        if scope is None:
            self._scoped.clear()
        else:
            self._scoped.pop(scope, None)
        self._hits = 0
        self._misses = 0

    def record_hit(self) -> None:
        self._hits += 1

    def record_miss(self) -> None:
        self._misses += 1

    def stats(self) -> dict[str, int]:
        bucket = self._bucket()
        return {
            "entries": len(bucket),
            "selector_cache_hits": self._hits,
            "selector_cache_misses": self._misses,
        }


_CACHE_SCOPE: ContextVar[str] = ContextVar("RANK_SELECTOR_CACHE_SCOPE", default="default")
_WINDOW_METRIC_CACHE = WindowMetricCache()

# Backwards compatibility counters exposed in tests.
selector_cache_hits = 0
selector_cache_misses = 0


@contextmanager
def selector_cache_scope(scope: str) -> Iterator[None]:
    """Context manager to isolate cache entries per run or request."""

    token = _CACHE_SCOPE.set(scope)
    try:
        yield
    finally:
        _CACHE_SCOPE.reset(token)


def set_window_metric_cache_limit(max_entries: int | None) -> int | None:
    """Set the selector cache size limit, returning the previous value."""

    previous = _WINDOW_METRIC_CACHE.set_limit(max_entries)
    _sync_cache_counters()
    return previous


def _cov_metric_from_payload(
    metric_name: str, payload: CovPayload, columns: Iterable[str]
) -> pd.Series:
    if metric_name == "__COV_VAR__":
        return pd.Series(payload.cov.diagonal(), index=columns, name="CovVar")
    diag = np.sqrt(np.clip(np.diag(payload.cov), 0, None))
    if diag.size <= 1:
        return pd.Series(0.0, index=columns, name="AvgCorr")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(diag, diag)
        corr = np.divide(payload.cov, denom, out=np.zeros_like(payload.cov), where=denom != 0)
    sums = corr.sum(axis=1) - 1.0
    avg = sums / (corr.shape[0] - 1)
    return pd.Series(avg, index=columns, name="AvgCorr")


def _compute_covariance_payload(
    bundle: WindowMetricBundle,
    cov_cache: CovCache | None,
    *,
    enable_cov_cache: bool,
    incremental_cov: bool,
) -> CovPayload:
    from ..perf.cache import compute_cov_payload

    if not enable_cov_cache or cov_cache is None:
        return compute_cov_payload(bundle.in_sample_df, materialise_aggregates=incremental_cov)
    key = cov_cache.make_key(
        bundle.start or "0000-00",
        bundle.end or "0000-00",
        bundle.in_sample_df.columns,
        bundle.freq,
    )
    return cov_cache.get_or_compute(
        key,
        lambda: compute_cov_payload(bundle.in_sample_df, materialise_aggregates=incremental_cov),
    )


_SELECTOR_CACHE_HITS = 0
_SELECTOR_CACHE_MISSES = 0


def _sync_cache_counters() -> None:
    """Mirror internal cache counters to public module attributes."""

    global selector_cache_hits, selector_cache_misses
    stats = _WINDOW_METRIC_CACHE.stats()
    selector_cache_hits = stats["selector_cache_hits"]
    selector_cache_misses = stats["selector_cache_misses"]
    _globals = globals()
    _globals["_SELECTOR_CACHE_HITS"] = selector_cache_hits
    _globals["_SELECTOR_CACHE_MISSES"] = selector_cache_misses


def make_window_key(
    start: str, end: str, universe: Iterable[str], stats_cfg: RiskStatsConfig
) -> WindowKey:
    """Return a stable cache key for a selector window."""

    canonical = _canonicalise_labels(universe)
    return (
        str(start),
        str(end),
        _hash_universe(canonical),
        _stats_cfg_hash(stats_cfg),
    )


def get_window_metric_bundle(window_key: WindowKey) -> WindowMetricBundle | None:
    """Return the cached bundle for *window_key* if present."""
    # Declare globals because we mutate (_SELECTOR_CACHE_HITS += 1) below.

    bundle = _WINDOW_METRIC_CACHE.get(window_key)
    if bundle is None:
        _sync_cache_counters()
        return None
    _sync_cache_counters()
    return bundle


def store_window_metric_bundle(window_key: WindowKey | None, bundle: WindowMetricBundle) -> None:
    """Store *bundle* under *window_key* when provided."""

    _WINDOW_METRIC_CACHE.put(window_key, bundle)


def clear_window_metric_cache(scope: str | None = None) -> None:
    """Reset the selector window cache and counters.

    When *scope* is provided, only the scoped cache entries are cleared. The
    instrumentation counters are reset in all cases.
    """

    _WINDOW_METRIC_CACHE.clear(scope)
    _sync_cache_counters()


def reset_selector_cache(scope: str | None = None) -> None:
    """Compatibility alias for :func:`clear_window_metric_cache`."""

    clear_window_metric_cache(scope)


def selector_cache_stats() -> dict[str, int]:
    """Return selector cache instrumentation counters."""

    stats = _WINDOW_METRIC_CACHE.stats()
    stats["selector_cache_hits"] = _SELECTOR_CACHE_HITS
    stats["selector_cache_misses"] = _SELECTOR_CACHE_MISSES
    return stats


# ──────────────────────────────────────────────────────────────────
# Metric transformer: raw | rank | percentile | zscore
# ──────────────────────────────────────────────────────────────────
def _apply_transform(
    series: pd.Series,
    *,
    mode: str = "raw",
    window: int | None = None,
    rank_pct: float | None = None,
    ddof: int = 0,
) -> pd.Series:
    """Return a transformed copy of *series* without mutating the original.

    Parameters
    ----------
    mode      : 'raw' | 'rank' | 'percentile' | 'zscore'
    window    : trailing periods for z‑score (ignored otherwise)
    rank_pct  : top‑X% mask when mode == 'percentile'
    ddof      : degrees of freedom for std in z‑score
    """
    if mode == "raw":
        return series

    if mode == "rank":
        return series.rank(ascending=False, pct=False)

    if mode == "percentile":
        if rank_pct is None:
            raise ValueError("rank_pct must be set for percentile transform")
        k = max(int(round(len(series) * rank_pct)), 1)
        mask = series.rank(ascending=False, pct=False) <= k
        return series.where(mask, np.nan)

    if mode == "zscore":
        if window is None or window > len(series):
            window = len(series)
        recent = series.iloc[-window:]
        mu = recent.mean()
        sigma = recent.std(ddof=ddof)
        # If variance is zero or std is non-finite, z-scores are undefined.
        # Return zeros to keep candidates available for ranking rather than
        # producing NaNs that get dropped upstream.
        if not np.isfinite(sigma) or sigma == 0:
            return pd.Series(0.0, index=series.index, dtype=float)
        return (series - mu) / sigma

    raise ValueError(f"unknown transform mode '{mode}'")


def rank_select_funds(
    df: pd.DataFrame,
    cfg: RiskStatsConfig,
    *,
    inclusion_approach: str = "top_n",
    n: int | None = None,
    pct: float | None = None,
    threshold: float | None = None,
    score_by: str = DEFAULT_METRIC,
    blended_weights: dict[str, float] | None = None,
    transform: str = "raw",
    transform_mode: str | None = None,
    zscore_window: int | None = None,
    zscore_ddof: int = 1,
    rank_pct: float = 0.5,
    limit_one_per_firm: bool = True,
    bundle: WindowMetricBundle | None = None,
    window_key: WindowKey | None = None,
    cov_cache: CovCache | None = None,
    freq: str = "M",
    enable_cov_cache: bool = True,
    incremental_cov: bool = False,
    store_bundle: bool = True,
    risk_free: float | pd.Series | None = None,
    return_diagnostics: bool = False,
) -> list[str] | tuple[list[str], RankSelectionDiagnostics | None]:
    """Select funds based on ranking by a specified metric."""

    diagnostics: RankSelectionDiagnostics | None = None

    def _record_diagnostics(
        reason: str,
        *,
        threshold_value: float | None = None,
        non_null_override: int | None = None,
    ) -> None:
        nonlocal diagnostics
        diagnostics = RankSelectionDiagnostics(
            reason=reason,
            metric=metric_name,
            transform=transform,
            inclusion_approach=inclusion_approach,
            total_candidates=len(universe),
            non_null_scores=(len(scores) if non_null_override is None else non_null_override),
            threshold=threshold_value,
        )
        warnings.warn(diagnostics.message(), RuntimeWarning)

    # Handle transform_mode alias
    if transform_mode is not None:
        transform = transform_mode

    metric_name = _METRIC_ALIASES.get(score_by, score_by)

    df = _ensure_canonical_columns(df)
    universe = tuple(df.columns)
    cfg_hash = _stats_cfg_hash(cfg)
    scores = pd.Series(dtype=float)

    if len(universe) == 0:
        _record_diagnostics(
            "No candidate columns available for ranking",
            non_null_override=0,
        )
        if return_diagnostics:
            return [], diagnostics
        return []

    if risk_free is not None:
        bundle = None
        window_key = None
        store_bundle = False
        enable_cov_cache = False
    if bundle is not None and bundle.universe != universe:
        raise ValueError("Provided bundle does not match DataFrame columns")
    if bundle is not None and bundle.stats_cfg_hash != cfg_hash:
        raise ValueError("Provided bundle does not match stats configuration")

    if bundle is None and window_key is not None:
        cached_bundle = get_window_metric_bundle(window_key)
        if cached_bundle is not None and cached_bundle.universe == universe:
            bundle = cached_bundle
        elif cached_bundle is not None and cached_bundle.universe != universe:
            bundle = None

    if bundle is None:
        metrics_frame = pd.DataFrame(index=universe, dtype=float)
        bundle = WindowMetricBundle(
            key=window_key,
            start=window_key[0] if window_key else "",
            end=window_key[1] if window_key else "",
            freq=freq,
            stats_cfg_hash=cfg_hash,
            universe=universe,
            in_sample_df=df.copy(),
            _metrics=metrics_frame,
        )
        if store_bundle:
            store_window_metric_bundle(window_key, bundle)
    elif bundle is not None:
        bundle.freq = freq

    # Compute metric scores
    if metric_name == "blended":
        if blended_weights is None:
            raise ValueError("blended score requires blended_weights parameter")
        target_df = bundle.in_sample_df if bundle is not None else df
        scores = blended_score(
            target_df,
            blended_weights,
            cfg,
            bundle=bundle,
            cov_cache=cov_cache,
            enable_cov_cache=enable_cov_cache,
            incremental_cov=incremental_cov,
            risk_free_override=risk_free,
        )
    else:
        if bundle is not None:
            scores = bundle.ensure_metric(
                metric_name,
                cfg,
                cov_cache=cov_cache,
                enable_cov_cache=enable_cov_cache,
                incremental_cov=incremental_cov,
            ).copy()
        else:
            scores = _call_metric_series(df, metric_name, cfg, risk_free_override=risk_free)

    # Apply transform
    scores = _apply_transform(
        scores,
        mode=transform,
        window=zscore_window,
        ddof=zscore_ddof,
        rank_pct=rank_pct,
    )
    # Determine sort order:
    # - transform == 'rank' produces 1=best so ascending True
    # - for metrics where smaller is better, sort ascending
    # - otherwise sort descending (larger is better)
    if transform == "rank":
        ascending = True
    else:
        ascending = metric_name in ASCENDING_METRICS

    # Drop NaNs (e.g., from percentile masking) before sorting
    scores = scores.dropna()
    if scores.empty:
        _record_diagnostics(
            "No candidate scores available after filtering and transform",
        )
        if return_diagnostics:
            return [], diagnostics
        return []
    scores = scores.sort_values(ascending=ascending)

    def _firm_key(name: str) -> str:
        # Normalize and derive a grouping key intended to capture the firm.
        # Heuristic:
        #  - tokenize on non-letters
        #  - if the first token seems like a brand/acronym (ALL CAPS or short), use it
        #  - otherwise use the first two tokens
        tokens = [t for t in _FIRM_NAME_TOKENIZER.split(str(name)) if t]
        if not tokens:
            return str(name).strip().lower()
        t0 = tokens[0]
        if t0.isupper() or len(t0) <= 3:
            return t0.lower()
        # Include second token when available to distinguish generic first words
        return (t0 + (" " + tokens[1] if len(tokens) > 1 else "")).lower()

    def _dedupe_by_firm(cands: list[str], k: int | None = None) -> list[str]:
        """Best-effort one-per-firm selection preserving order.

        If ``k`` is provided and unique firms are insufficient to reach ``k``,
        backfill with remaining candidates (even if from the same firm) to
        satisfy the requested count. When ``k`` is ``None`` (e.g., threshold),
        only uniqueness is enforced.
        """
        if not limit_one_per_firm:
            return cands if k is None else cands[:k]

        chosen: list[str] = []
        seen: set[str] = set()

        # First pass: enforce uniqueness by firm
        for name in cands:
            fk = _firm_key(name)
            if fk in seen:
                continue
            seen.add(fk)
            chosen.append(name)
            if k is not None and len(chosen) >= k:
                break

        # If we have a target count and didn't reach it, backfill
        if k is not None and len(chosen) < k:
            for name in cands:
                if name in chosen:
                    continue
                chosen.append(name)
                if len(chosen) >= k:
                    break
        return chosen

    # Apply inclusion approach
    if inclusion_approach == "top_n":
        if n is None:
            raise ValueError("top_n requires parameter n")
        ordered_top_n: list[str] = [str(x) for x in scores.index.tolist()]
        selection = _dedupe_by_firm(ordered_top_n, k=n)
    elif inclusion_approach == "top_pct":
        if pct is None or not 0 < pct <= 1:
            raise ValueError("top_pct requires 0 < pct <= 1")
        k = max(1, int(round(len(scores) * pct)))
        ordered_top_pct: list[str] = [str(x) for x in scores.index.tolist()]
        selection = _dedupe_by_firm(ordered_top_pct, k=k)
    elif inclusion_approach == "threshold":
        if threshold is None:
            raise ValueError("threshold approach requires parameter threshold")
        # For ascending=True (smaller is better), keep scores <= threshold
        # else keep scores >= threshold
        mask = scores <= threshold if ascending else scores >= threshold
        filtered_scores = scores[mask]
        if filtered_scores.empty:
            _record_diagnostics(
                "All candidate scores filtered out by threshold",
                threshold_value=threshold,
                non_null_override=0,
            )
            if return_diagnostics:
                return [], diagnostics
            return []
        ordered_threshold: list[str] = [str(x) for x in filtered_scores.index.tolist()]
        selection = _dedupe_by_firm(ordered_threshold)
    else:
        raise ValueError("Unknown inclusion_approach")

    if return_diagnostics:
        return selection, diagnostics
    return selection


@dataclass
class RiskStatsConfig:
    """Metrics and risk free configuration."""

    metrics_to_run: list[str] = field(
        default_factory=lambda: [
            "AnnualReturn",
            "Volatility",
            "Sharpe",
            "Sortino",
            "MaxDrawdown",
            "InformationRatio",
        ]
    )
    risk_free: float = 0.0
    periods_per_year: int = 12


METRIC_REGISTRY: dict[str, Callable[..., float | pd.Series]] = {}
_METRIC_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "_TREND_METRIC_CONTEXT", default=None
)

# Map snake_case config names to the canonical registry keys.
_METRIC_ALIASES: dict[str, str] = {
    "annual_return": "AnnualReturn",
    "volatility": "Volatility",
    "sharpe_ratio": "Sharpe",
    "sharpe": "Sharpe",  # Add lowercase sharpe alias
    "sortino_ratio": "Sortino",
    "max_drawdown": "MaxDrawdown",
    "information_ratio": "InformationRatio",
    "avg_corr": "AvgCorr",
}


def canonical_metric_list(names: Iterable[str] | None = None) -> list[str]:
    """Return registry keys normalised from ``names``, or all registered
    metrics if names is None."""
    if names is None:
        return list(METRIC_REGISTRY.keys())
    result = []
    for n in names:
        result.append(_METRIC_ALIASES.get(n, n))
    return result


def register_metric(
    name: str,
) -> Callable[
    [Callable[..., float | pd.Series]],
    Callable[..., float | pd.Series],
]:
    """Register ``fn`` under ``name`` in :data:`METRIC_REGISTRY`."""

    def decorator(
        fn: Callable[..., float | pd.Series],
    ) -> Callable[..., float | pd.Series]:
        METRIC_REGISTRY[name] = fn
        return fn

    return decorator


def _get_metric_context() -> dict[str, Any]:
    ctx = _METRIC_CONTEXT.get()
    if ctx is None:
        raise RuntimeError("Metric context is unavailable for frame-aware metrics")
    return ctx


# Register basic metrics from the public ``metrics`` module
register_metric("AnnualReturn")(
    lambda s, *, periods_per_year=12, **k: _metrics.annual_return(
        s, periods_per_year=periods_per_year
    )
)

register_metric("Volatility")(
    lambda s, *, periods_per_year=12, **k: _metrics.volatility(s, periods_per_year=periods_per_year)
)

register_metric("Sharpe")(
    lambda s, *, periods_per_year=12, risk_free=0.0: _metrics.sharpe_ratio(
        s,
        periods_per_year=periods_per_year,
        risk_free=risk_free,
    )
)

# Register lowercase 'sharpe' for compatibility
register_metric("sharpe")(
    lambda s, *, periods_per_year=12, risk_free=0.0: _metrics.sharpe_ratio(
        s,
        periods_per_year=periods_per_year,
        risk_free=risk_free,
    )
)

register_metric("Sortino")(
    lambda s, *, periods_per_year=12, risk_free=0.0, target=0.0, **k: _metrics.sortino_ratio(
        s,
        periods_per_year=periods_per_year,
        # Align with pipeline usage: treat the provided risk-free override as
        # the target return series/rate for Sortino.
        target=risk_free if risk_free is not None else target,
    )
)

register_metric("MaxDrawdown")(lambda s, **k: _metrics.max_drawdown(s))

register_metric("InformationRatio")(
    lambda s, *, periods_per_year=12, risk_free=0.0, benchmark=None, **k: _metrics.information_ratio(
        s,
        # Align with pipeline usage: treat the provided risk-free override as
        # the benchmark for IR when no explicit benchmark series is provided.
        benchmark=(
            benchmark
            if benchmark is not None
            else (risk_free if risk_free is not None else pd.Series(0, index=s.index))
        ),
        periods_per_year=periods_per_year,
    )
)


# Average correlation metric (frame-aware via metric context)
@register_metric("AvgCorr")
def _avg_corr_metric(series: pd.Series, **_: Any) -> float:
    """Fallback AvgCorr implementation using the metric context frame."""

    ctx = _get_metric_context()
    frame = ctx.get("frame")
    if frame is None or frame.empty:
        return 0.0
    if frame.shape[1] <= 1:
        return 0.0
    corr = ctx.get("avg_corr_corr")
    corr_frame_id = ctx.get("avg_corr_frame_id")
    if corr is None or corr_frame_id != id(frame):
        corr = frame.corr(method="pearson", min_periods=2)
        ctx["avg_corr_corr"] = corr
        ctx["avg_corr_frame_id"] = id(frame)
    col = series.name
    if col not in corr.columns:
        return 0.0
    col_corr = corr.loc[col].drop(labels=[col], errors="ignore").dropna()
    if col_corr.empty:
        return 0.0
    return float(col_corr.mean())


# ===============================================================
#  NEW: RANK‑BASED FUND SELECTION
# ===============================================================

ASCENDING_METRICS = {"MaxDrawdown"}  # smaller is better
DEFAULT_METRIC = "AnnualReturn"


def _compute_metric_series(
    in_sample_df: pd.DataFrame,
    metric_name: str,
    stats_cfg: RiskStatsConfig,
    *,
    risk_free_override: float | pd.Series | None = None,
) -> pd.Series:
    """Return a pd.Series (index = fund code, value = metric score).

    Vectorised: uses the registered metric on each column.
    """
    fn = METRIC_REGISTRY.get(metric_name)
    if fn is None:
        raise ValueError(f"Metric '{metric_name}' not registered")
    context: dict[str, Any] = {"frame": in_sample_df}
    token = _METRIC_CONTEXT.set(context)
    try:
        rf_value: float | pd.Series = (
            risk_free_override if risk_free_override is not None else stats_cfg.risk_free
        )
        return in_sample_df.apply(
            fn,
            periods_per_year=stats_cfg.periods_per_year,
            risk_free=rf_value,
            axis=0,
        )
    finally:
        _METRIC_CONTEXT.reset(token)


@cache
def _metric_fn_accepts_risk_free_override(func: Callable[..., Any]) -> bool:
    """Return True if *func* accepts ``risk_free_override`` keyword."""

    try:
        return "risk_free_override" in inspect.signature(func).parameters
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return False


def _call_metric_series(
    in_sample_df: pd.DataFrame,
    metric_name: str,
    stats_cfg: RiskStatsConfig,
    *,
    risk_free_override: float | pd.Series | None = None,
) -> pd.Series:
    """Invoke :func:`_compute_metric_series` with optional RF override.

    Tests frequently monkeypatch ``_compute_metric_series`` with simplified
    stand-ins that do not accept ``risk_free_override``.  This helper inspects
    the active callable at runtime and only forwards the override when it is
    supported, preserving backwards compatibility.
    """

    fn = _compute_metric_series
    if risk_free_override is not None and _metric_fn_accepts_risk_free_override(fn):
        return fn(
            in_sample_df,
            metric_name,
            stats_cfg,
            risk_free_override=risk_free_override,
        )
    return fn(in_sample_df, metric_name, stats_cfg)


def _ensure_cov_payload(
    in_sample_df: pd.DataFrame, bundle: WindowMetricBundle | None
) -> CovPayload:
    """Return a covariance payload, populating the bundle if provided."""

    if bundle is not None and bundle.cov_payload is not None:
        return bundle.cov_payload

    from ..perf.cache import compute_cov_payload

    payload = compute_cov_payload(in_sample_df)
    if bundle is not None:
        bundle.cov_payload = payload
    return payload


def _metric_from_cov_payload(
    metric_name: str, in_sample_df: pd.DataFrame, payload: CovPayload
) -> pd.Series:
    """Compute covariance-derived metric series from ``payload``."""

    if metric_name == "__COV_VAR__":
        return pd.Series(payload.cov.diagonal(), index=in_sample_df.columns, name="CovVar")

    diag = np.sqrt(np.clip(np.diag(payload.cov), 0, None))
    if diag.size <= 1:
        return pd.Series(0.0, index=in_sample_df.columns, name="AvgCorr")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(diag, diag)
        corr = np.divide(payload.cov, denom, out=np.zeros_like(payload.cov), where=denom != 0)
    sums = corr.sum(axis=1) - 1.0
    avg = sums / (corr.shape[0] - 1)
    return pd.Series(avg, index=in_sample_df.columns, name="AvgCorr")


def compute_metric_series_with_cache(
    in_sample_df: pd.DataFrame,
    metric_name: str,
    stats_cfg: RiskStatsConfig,
    *,
    risk_free_override: float | pd.Series | None = None,
    cov_cache: CovCache | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
    freq: str = "M",
    enable_cache: bool = True,
    incremental_cov: bool = False,
) -> pd.Series:
    """Compute metric series with optional covariance caching.

    Current pipeline calls remain unchanged; this helper is opt-in for
    metrics that require a covariance matrix. A synthetic metric name
    ``__COV_VAR__`` demonstrates usage by returning diagonal variances
    from a cached covariance payload. Real metrics can hook into this
    path without altering existing registry semantics.
    """
    if metric_name not in {"__COV_VAR__", "AvgCorr"}:
        return _call_metric_series(
            in_sample_df,
            metric_name,
            stats_cfg,
            risk_free_override=risk_free_override,
        )
    from ..perf.cache import compute_cov_payload

    # Caching disabled path
    if (cov_cache is None) or (not enable_cache):
        payload = compute_cov_payload(in_sample_df)
    else:
        key = cov_cache.make_key(
            window_start or "0000-00",
            window_end or "0000-00",
            in_sample_df.columns,
            freq,
        )
        # NOTE: incremental_cov is a future optimization: requires caller to
        # provide previous payload & row deltas. For now we simply ignore the
        # flag and rely on standard cache lookups. Hook point documented.
        payload = cov_cache.get_or_compute(
            key,
            lambda: compute_cov_payload(in_sample_df, materialise_aggregates=incremental_cov),
        )
    if metric_name == "__COV_VAR__":
        return pd.Series(payload.cov.diagonal(), index=in_sample_df.columns, name="CovVar")
    # AvgCorr computation
    diag = np.sqrt(np.clip(np.diag(payload.cov), 0, None))
    if diag.size <= 1:
        return pd.Series(0.0, index=in_sample_df.columns, name="AvgCorr")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(diag, diag)
        corr = np.divide(payload.cov, denom, out=np.zeros_like(payload.cov), where=denom != 0)
    sums = corr.sum(axis=1) - 1.0
    avg = sums / (corr.shape[0] - 1)
    return pd.Series(avg, index=in_sample_df.columns, name="AvgCorr")


def _zscore(series: pd.Series) -> pd.Series:
    """Return z‑scores (mean 0, stdev 1).

    Gracefully handles zero σ.
    """
    μ, σ = series.mean(), series.std(ddof=0)
    if σ == 0:
        return pd.Series(0.0, index=series.index)
    return (series - μ) / σ


def blended_score(
    in_sample_df: pd.DataFrame,
    weights: dict[str, float],
    stats_cfg: RiskStatsConfig,
    *,
    bundle: WindowMetricBundle | None = None,
    cov_cache: CovCache | None = None,
    enable_cov_cache: bool = True,
    incremental_cov: bool = False,
    risk_free_override: float | pd.Series | None = None,
) -> pd.Series:
    """Z‑score each contributing metric, then weighted linear combo."""
    if not weights:
        raise ValueError("blended_score requires non-empty weights dict")
    # Normalize metric names using _METRIC_ALIASES
    canonical_weights: dict[str, float] = {}
    for k, v in weights.items():
        canonical = _METRIC_ALIASES.get(k, k)
        if canonical in canonical_weights:
            canonical_weights[canonical] += v
        else:
            canonical_weights[canonical] = v
    total = sum(canonical_weights.values())
    if total == 0:
        raise ValueError("Sum of weights must not be zero")
    w_norm = {k: v / total for k, v in canonical_weights.items()}

    combo = pd.Series(0.0, index=in_sample_df.columns)
    for metric, w in w_norm.items():
        if bundle is not None:
            raw = bundle.ensure_metric(
                metric,
                stats_cfg,
                cov_cache=cov_cache,
                enable_cov_cache=enable_cov_cache,
                incremental_cov=incremental_cov,
            )
        else:
            raw = _call_metric_series(
                in_sample_df,
                metric,
                stats_cfg,
                risk_free_override=risk_free_override,
            )
        z = _zscore(raw)
        # If metric is "smaller‑is‑better", *invert* before z‑score
        if metric in ASCENDING_METRICS:
            z *= -1
        combo += w * z
    return combo


# ===============================================================
#  UI SCAFFOLD (lazy-loaded to keep optional widget deps)
# ===============================================================


def build_ui() -> Any:
    """Load the optional notebook UI helpers for ranking workflows."""

    if importlib.util.find_spec("ipywidgets") is None:
        raise ImportError("ipywidgets is required for the ranking UI helpers.")
    module = importlib.import_module("trend_analysis.ui.rank_widgets")
    return module.build_ui()


__all__ = [
    "RiskStatsConfig",
    "register_metric",
    "METRIC_REGISTRY",
    "WindowMetricBundle",
    "make_window_key",
    "get_window_metric_bundle",
    "reset_selector_cache",
    "selector_cache_hits",
    "selector_cache_misses",
    "blended_score",
    "_call_metric_series",
    "compute_metric_series_with_cache",
    "rank_select_funds",
    "selector_cache_stats",
    "clear_window_metric_cache",
    "selector_cache_scope",
    "set_window_metric_cache_limit",
    "build_ui",
    "canonical_metric_list",
]
