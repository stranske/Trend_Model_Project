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
import inspect
import io
import json
import re
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, cast

import ipywidgets as widgets
import numpy as np
import pandas as pd

from .. import metrics as _metrics
from ..data import ensure_datetime, load_csv
from ..export import Formatter

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..perf.cache import CovCache, CovPayload

# Compiled regex pattern for better performance when processing large files
_FIRM_NAME_TOKENIZER = re.compile(r"[^A-Za-z]+")

DEFAULT_METRIC = "annual_return"

WindowKey = tuple[str, str, str, str]


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


def _stats_cfg_hash(cfg: "RiskStatsConfig") -> str:
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
    cov_payload: "CovPayload | None" = None

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
        stats_cfg: "RiskStatsConfig",
        *,
        cov_cache: "CovCache | None" = None,
        enable_cov_cache: bool = True,
        incremental_cov: bool = False,
    ) -> pd.Series:
        """Ensure *metric_name* exists in the cached frame and return it."""

        global _SELECTOR_CACHE_HITS, _SELECTOR_CACHE_MISSES
        canonical = _METRIC_ALIASES.get(metric_name, metric_name)
        if canonical in self._metrics.columns:
            _SELECTOR_CACHE_HITS += 1
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
            series = _cov_metric_from_payload(
                canonical, payload, self.in_sample_df.columns
            )
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
                compute=lambda: _compute_metric_series(
                    self.in_sample_df, canonical, stats_cfg
                ),
                enable=use_metric_cache,
                cache=None,
            )
        series = series.astype(float)
        self._metrics[canonical] = series
        _SELECTOR_CACHE_MISSES += 1
        _sync_cache_counters()
        return self._metrics[canonical]


def _cov_metric_from_payload(
    metric_name: str, payload: "CovPayload", columns: Iterable[str]
) -> pd.Series:
    if metric_name == "__COV_VAR__":
        return pd.Series(payload.cov.diagonal(), index=columns, name="CovVar")
    diag = np.sqrt(np.clip(np.diag(payload.cov), 0, None))
    if diag.size <= 1:
        return pd.Series(0.0, index=columns, name="AvgCorr")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(diag, diag)
        corr = np.divide(
            payload.cov, denom, out=np.zeros_like(payload.cov), where=denom != 0
        )
    sums = corr.sum(axis=1) - 1.0
    avg = sums / (corr.shape[0] - 1)
    return pd.Series(avg, index=columns, name="AvgCorr")


def _compute_covariance_payload(
    bundle: WindowMetricBundle,
    cov_cache: "CovCache | None",
    *,
    enable_cov_cache: bool,
    incremental_cov: bool,
) -> "CovPayload":
    from ..perf.cache import compute_cov_payload

    if not enable_cov_cache or cov_cache is None:
        return compute_cov_payload(
            bundle.in_sample_df, materialise_aggregates=incremental_cov
        )
    key = cov_cache.make_key(
        bundle.start or "0000-00",
        bundle.end or "0000-00",
        bundle.in_sample_df.columns,
        bundle.freq,
    )
    return cov_cache.get_or_compute(
        key,
        lambda: compute_cov_payload(
            bundle.in_sample_df, materialise_aggregates=incremental_cov
        ),
    )


_WINDOW_METRIC_BUNDLES: dict[WindowKey, WindowMetricBundle] = {}
_SELECTOR_CACHE_HITS = 0
_SELECTOR_CACHE_MISSES = 0

# Backwards compatibility counters exposed in tests.
selector_cache_hits = 0
selector_cache_misses = 0


def _sync_cache_counters() -> None:
    """Mirror internal cache counters to public module attributes."""

    global selector_cache_hits, selector_cache_misses
    selector_cache_hits = _SELECTOR_CACHE_HITS
    selector_cache_misses = _SELECTOR_CACHE_MISSES


def make_window_key(
    start: str, end: str, universe: Iterable[str], stats_cfg: "RiskStatsConfig"
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
    global _SELECTOR_CACHE_HITS

    bundle = _WINDOW_METRIC_BUNDLES.get(window_key)
    if bundle is None:
        return None
    _SELECTOR_CACHE_HITS += 1
    _sync_cache_counters()
    return bundle


def store_window_metric_bundle(
    window_key: WindowKey | None, bundle: WindowMetricBundle
) -> None:
    """Store *bundle* under *window_key* when provided."""

    if window_key is None:
        return
    _WINDOW_METRIC_BUNDLES[window_key] = bundle


def clear_window_metric_cache() -> None:
    """Reset the selector window cache and counters."""

    global _SELECTOR_CACHE_HITS, _SELECTOR_CACHE_MISSES

    _WINDOW_METRIC_BUNDLES.clear()
    _SELECTOR_CACHE_HITS = 0
    _SELECTOR_CACHE_MISSES = 0
    _sync_cache_counters()


def reset_selector_cache() -> None:
    """Compatibility alias for :func:`clear_window_metric_cache`."""

    clear_window_metric_cache()


def selector_cache_stats() -> dict[str, int]:
    """Return selector cache instrumentation counters."""

    return {
        "entries": len(_WINDOW_METRIC_BUNDLES),
        "selector_cache_hits": _SELECTOR_CACHE_HITS,
        "selector_cache_misses": _SELECTOR_CACHE_MISSES,
    }


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
    cov_cache: "CovCache | None" = None,
    freq: str = "M",
    enable_cov_cache: bool = True,
    incremental_cov: bool = False,
    store_bundle: bool = True,
    risk_free: float | pd.Series | None = None,
) -> list[str]:
    """Select funds based on ranking by a specified metric."""

    # Handle transform_mode alias
    if transform_mode is not None:
        transform = transform_mode

    metric_name = _METRIC_ALIASES.get(score_by, score_by)

    df = _ensure_canonical_columns(df)
    universe = tuple(df.columns)
    cfg_hash = _stats_cfg_hash(cfg)

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
            scores = _call_metric_series(
                df, metric_name, cfg, risk_free_override=risk_free
            )

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
        return _dedupe_by_firm(ordered_top_n, k=n)
    elif inclusion_approach == "top_pct":
        if pct is None or not 0 < pct <= 1:
            raise ValueError("top_pct requires 0 < pct <= 1")
        k = max(1, int(round(len(scores) * pct)))
        ordered_top_pct: list[str] = [str(x) for x in scores.index.tolist()]
        return _dedupe_by_firm(ordered_top_pct, k=k)
    elif inclusion_approach == "threshold":
        if threshold is None:
            raise ValueError("threshold approach requires parameter threshold")
        # For ascending=True (smaller is better), keep scores <= threshold
        # else keep scores >= threshold
        mask = scores <= threshold if ascending else scores >= threshold
        ordered_threshold: list[str] = [str(x) for x in scores[mask].index.tolist()]
        return _dedupe_by_firm(ordered_threshold)
    else:
        raise ValueError("Unknown inclusion_approach")


@dataclass
class RiskStatsConfig:
    """Metrics and risk free configuration."""

    metrics_to_run: List[str] = field(
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


METRIC_REGISTRY: Dict[str, Callable[..., float | pd.Series]] = {}
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
    lambda s, *, periods_per_year=12, **k: _metrics.volatility(
        s, periods_per_year=periods_per_year
    )
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
    lambda s, *, periods_per_year=12, target=0.0, **k: _metrics.sortino_ratio(
        s,
        periods_per_year=periods_per_year,
        target=target,
    )
)

register_metric("MaxDrawdown")(lambda s, **k: _metrics.max_drawdown(s))

register_metric("InformationRatio")(
    lambda s, *, periods_per_year=12, benchmark=None, **k: _metrics.information_ratio(
        s,
        benchmark=benchmark if benchmark is not None else pd.Series(0, index=s.index),
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
            risk_free_override
            if risk_free_override is not None
            else stats_cfg.risk_free
        )
        return in_sample_df.apply(
            fn,
            periods_per_year=stats_cfg.periods_per_year,
            risk_free=rf_value,
            axis=0,
        )
    finally:
        _METRIC_CONTEXT.reset(token)


@lru_cache(maxsize=None)
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
) -> "CovPayload":
    """Return a covariance payload, populating the bundle if provided."""

    if bundle is not None and bundle.cov_payload is not None:
        return bundle.cov_payload

    from ..perf.cache import compute_cov_payload

    payload = compute_cov_payload(in_sample_df)
    if bundle is not None:
        bundle.cov_payload = payload
    return payload


def _metric_from_cov_payload(
    metric_name: str, in_sample_df: pd.DataFrame, payload: "CovPayload"
) -> pd.Series:
    """Compute covariance-derived metric series from ``payload``."""

    if metric_name == "__COV_VAR__":
        return pd.Series(
            payload.cov.diagonal(), index=in_sample_df.columns, name="CovVar"
        )

    diag = np.sqrt(np.clip(np.diag(payload.cov), 0, None))
    if diag.size <= 1:
        return pd.Series(0.0, index=in_sample_df.columns, name="AvgCorr")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(diag, diag)
        corr = np.divide(
            payload.cov, denom, out=np.zeros_like(payload.cov), where=denom != 0
        )
    sums = corr.sum(axis=1) - 1.0
    avg = sums / (corr.shape[0] - 1)
    return pd.Series(avg, index=in_sample_df.columns, name="AvgCorr")


def compute_metric_series_with_cache(
    in_sample_df: pd.DataFrame,
    metric_name: str,
    stats_cfg: RiskStatsConfig,
    *,
    risk_free_override: float | pd.Series | None = None,
    cov_cache: "CovCache | None" = None,
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
            lambda: compute_cov_payload(
                in_sample_df, materialise_aggregates=incremental_cov
            ),
        )
    if metric_name == "__COV_VAR__":
        return pd.Series(
            payload.cov.diagonal(), index=in_sample_df.columns, name="CovVar"
        )
    # AvgCorr computation
    diag = np.sqrt(np.clip(np.diag(payload.cov), 0, None))
    if diag.size <= 1:
        return pd.Series(0.0, index=in_sample_df.columns, name="AvgCorr")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(diag, diag)
        corr = np.divide(
            payload.cov, denom, out=np.zeros_like(payload.cov), where=denom != 0
        )
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
    cov_cache: "CovCache | None" = None,
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
#  UI SCAFFOLD (very condensed – Codex expands)
# ===============================================================


def build_ui() -> widgets.VBox:  # pragma: no cover - UI wiring exercised manually
    # -------------------- Step 1: data source & periods --------------------
    source_tb = widgets.ToggleButtons(
        options=["Path/URL", "Browse"],
        description="Source:",
    )
    csv_path = widgets.Text(description="CSV or URL:")
    file_up = widgets.FileUpload(accept=".csv", multiple=False)
    file_up.layout.display = "none"
    load_btn = widgets.Button(description="Load CSV", button_style="success")
    load_out = widgets.Output()

    in_start = widgets.Text(description="In Start:")
    in_end = widgets.Text(description="In End:")
    out_start = widgets.Text(description="Out Start:")
    out_end = widgets.Text(description="Out End:")

    session: dict[str, Any] = {"df": None, "rf": None}
    idx_select = widgets.SelectMultiple(options=[], description="Indices:")
    idx_select.layout.display = "none"
    bench_select = widgets.SelectMultiple(options=[], description="Benchmarks:")
    bench_select.layout.display = "none"
    step1_box = widgets.VBox(
        [
            source_tb,
            csv_path,
            file_up,
            load_btn,
            load_out,
            idx_select,
            bench_select,
            in_start,
            in_end,
            out_start,
            out_end,
        ]
    )

    def _load_action(_btn: widgets.Button) -> None:
        with load_out:
            load_out.clear_output()
            try:
                df: pd.DataFrame | None = None
                if source_tb.value == "Browse":
                    if not file_up.value:
                        print("Upload a CSV")
                        return
                    # ipywidgets 7.x returns a dict; 8.x returns a tuple
                    if isinstance(file_up.value, dict):
                        item = next(iter(file_up.value.values()))
                    else:
                        item = file_up.value[0]
                    df = pd.read_csv(io.BytesIO(item["content"]))
                else:
                    path = csv_path.value.strip()
                    if not path:
                        print("Enter CSV path or URL")
                        return
                    if path.startswith("http://") or path.startswith("https://"):
                        df = pd.read_csv(path)
                    else:
                        df = load_csv(path)
                if df is None:
                    print("Failed to load")
                    return
                df = ensure_datetime(df)
                session["df"] = df
                # session["rf"] = rf  # rf is not defined here, skip or set to None
                dates = df["Date"].dt.to_period("M")
                in_start.value = str(dates.min())
                in_end.value = str(dates.min() + 2)
                out_start.value = str(dates.min() + 3)
                out_end.value = str(dates.min() + 5)
                idx_select.options = [c for c in df.columns if c not in {"Date"}]
                idx_select.layout.display = "flex"
                bench_select.options = [c for c in df.columns if c not in {"Date"}]
                bench_select.layout.display = "flex"
                print(f"Loaded {len(df):,} rows")
            except Exception as exc:
                session["df"] = None
                print("Error:", exc)

    load_btn.on_click(_load_action)

    def _source_toggle(*_: Any) -> None:
        if source_tb.value == "Browse":
            file_up.layout.display = "flex"
            csv_path.layout.display = "none"
        else:
            file_up.layout.display = "none"
            csv_path.layout.display = "flex"

    source_tb.observe(_source_toggle, "value")
    _source_toggle()

    # -------------------- Step 2: selection & ranking ----------------------
    mode_dd = widgets.Dropdown(
        options=["all", "random", "manual", "rank"], description="Mode:"
    )
    random_n_int = widgets.BoundedIntText(value=8, min=1, description="Random N:")
    random_n_int.layout.display = "none"
    vol_ck = widgets.Checkbox(value=True, description="Vol‑adjust?")
    target_vol = widgets.BoundedFloatText(
        value=0.10, min=0.0, max=10.0, step=0.01, description="Target Vol:"
    )
    target_vol.layout.display = "none"
    use_rank_ck = widgets.Checkbox(value=False, description="Apply ranking?")
    next_btn_1 = widgets.Button(description="Next")

    # step‑2 widgets
    incl_dd = widgets.Dropdown(
        options=["top_n", "top_pct", "threshold"], description="Approach:"
    )
    metric_dd = widgets.Dropdown(
        options=list(METRIC_REGISTRY) + ["blended"], description="Score by:"
    )
    topn_int = widgets.BoundedIntText(value=10, min=1, description="N:")
    pct_flt = widgets.BoundedFloatText(
        value=0.10, min=0.01, max=1.0, step=0.01, description="Pct:"
    )
    thresh_f = widgets.FloatText(value=1.0, description="Threshold:")

    # blended area
    m1_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M1")
    w1_sl = widgets.FloatSlider(value=0.33, min=0, max=1.0, step=0.01)
    m2_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M2")
    w2_sl = widgets.FloatSlider(value=0.33, min=0, max=1.0, step=0.01)
    m3_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M3")
    w3_sl = widgets.FloatSlider(value=0.34, min=0, max=1.0, step=0.01)

    out_fmt = widgets.Dropdown(options=["excel", "csv", "json"], description="Output:")

    # --------------------------------------------------------------
    #  Callbacks and execution wiring
    # --------------------------------------------------------------

    blended_box = widgets.VBox([m1_dd, w1_sl, m2_dd, w2_sl, m3_dd, w3_sl])
    blended_box.layout.display = "none"

    rank_box = widgets.VBox(
        [incl_dd, metric_dd, topn_int, pct_flt, thresh_f, blended_box]
    )
    rank_box.layout.display = "none"

    def _update_random_vis(*_: Any) -> None:
        show = rank_unlocked and mode_dd.value == "random"
        random_n_int.layout.display = "flex" if show else "none"

    def _update_target_vol(*_: Any) -> None:
        target_vol.layout.display = "flex" if vol_ck.value else "none"

    manual_box = widgets.VBox()
    manual_box.layout.display = "none"
    manual_scores_html = widgets.HTML()
    manual_checks: list[widgets.Checkbox] = []
    manual_weights: list[widgets.FloatText] = []
    manual_total_lbl = widgets.Label("Total weight: 0 %")

    # track whether the user progressed past the first step
    rank_unlocked = False

    run_btn = widgets.Button(description="Run")
    output = widgets.Output()

    def _next_action(_: Any) -> None:
        nonlocal rank_unlocked
        if session["df"] is None:
            with load_out:
                load_out.clear_output()
                print("Load data first")
            return
        rank_unlocked = not rank_unlocked
        next_btn_1.layout.display = "none"
        _update_rank_vis()
        _update_random_vis()
        _update_manual()

    def _update_rank_vis(*_: Any) -> None:
        show = (
            rank_unlocked
            and session["df"] is not None
            and (mode_dd.value == "rank" or use_rank_ck.value)
        )
        rank_box.layout.display = "flex" if show else "none"
        _update_blended_vis()
        _update_manual()

    def _update_blended_vis(*_: Any) -> None:
        show = (
            rank_unlocked
            and metric_dd.value == "blended"
            and (mode_dd.value == "rank" or use_rank_ck.value)
        )
        blended_box.layout.display = "flex" if show else "none"

    def _update_manual(*_: Any) -> None:
        if mode_dd.value != "manual" or not rank_unlocked:
            manual_box.layout.display = "none"
            return
        df = session.get("df")
        if df is None:
            manual_box.children = [widgets.Label("Load data first")]
            manual_box.layout.display = "flex"
            return

        rf = session.get("rf", "RF")
        date_col = "Date"
        funds_all = [c for c in df.columns if c not in {date_col, rf}]
        try:

            def _to_dt(s: str) -> pd.Timestamp:
                return pd.to_datetime(f"{s}-01") + pd.offsets.MonthEnd(0)

            in_start_dt = _to_dt(in_start.value)
            in_end_dt = _to_dt(in_end.value)
            out_start_dt = _to_dt(out_start.value)
            out_end_dt = _to_dt(out_end.value)

            in_df = df[(df[date_col] >= in_start_dt) & (df[date_col] <= in_end_dt)]
            out_df = df[(df[date_col] >= out_start_dt) & (df[date_col] <= out_end_dt)]
            in_ok = ~in_df[funds_all].isna().any()
            out_ok = ~out_df[funds_all].isna().any()
            funds = [c for c in funds_all if in_ok[c] and out_ok[c]]
        except Exception:
            funds = funds_all

        # compute score frame for display
        try:
            from .. import pipeline

            sf = pipeline.single_period_run(
                df[[date_col] + funds],
                in_start.value,
                in_end.value,
            )
            manual_scores_html.value = sf.to_html(float_format="%.4f")
        except Exception:
            manual_scores_html.value = ""

        manual_checks.clear()
        manual_weights.clear()

        def _update_total(*_: Any) -> None:
            tot = sum(
                wt.value for chk, wt in zip(manual_checks, manual_weights) if chk.value
            )
            manual_total_lbl.value = f"Total weight: {tot:.0f} %"

        rows = []
        for f in funds:
            chk = widgets.Checkbox(value=False, description=f)
            wt = widgets.FloatText(value=0.0, layout=widgets.Layout(width="80px"))
            chk.observe(_update_total, "value")
            wt.observe(_update_total, "value")
            manual_checks.append(chk)
            manual_weights.append(wt)
            rows.append(widgets.HBox([chk, wt]))
        manual_box.children = [manual_scores_html] + rows + [manual_total_lbl]
        manual_box.layout.display = "flex"
        _update_total()

    def _update_inclusion_fields(*_: Any) -> None:
        topn_int.layout.display = "flex" if incl_dd.value == "top_n" else "none"
        pct_flt.layout.display = "flex" if incl_dd.value == "top_pct" else "none"
        thresh_f.layout.display = "flex" if incl_dd.value == "threshold" else "none"

    next_btn_1.on_click(_next_action)
    mode_dd.observe(_update_rank_vis, "value")
    mode_dd.observe(_update_random_vis, "value")
    use_rank_ck.observe(_update_rank_vis, "value")
    metric_dd.observe(_update_blended_vis, "value")
    incl_dd.observe(_update_inclusion_fields, "value")
    mode_dd.observe(_update_manual, "value")
    vol_ck.observe(_update_target_vol, "value")

    def _run_action(_btn: widgets.Button) -> None:
        rank_kwargs: dict[str, Any] | None = None
        if mode_dd.value == "rank" or use_rank_ck.value:
            rank_kwargs = {
                "inclusion_approach": incl_dd.value,
                "score_by": metric_dd.value,
            }
            if incl_dd.value == "top_n":
                rank_kwargs["n"] = int(topn_int.value)
            elif incl_dd.value == "top_pct":
                rank_kwargs["pct"] = float(pct_flt.value)
            elif incl_dd.value == "threshold":
                rank_kwargs["threshold"] = float(thresh_f.value)
            if metric_dd.value == "blended":
                rank_kwargs["blended_weights"] = {
                    m1_dd.value: w1_sl.value,
                    m2_dd.value: w2_sl.value,
                    m3_dd.value: w3_sl.value,
                }

        manual_funds: list[str] | None = None
        custom_weights: dict[str, float] | None = None
        if mode_dd.value == "manual":
            manual_funds = []
            custom_weights = {}
            for chk, wt in zip(manual_checks, manual_weights):
                if chk.value:
                    manual_funds.append(chk.description)
                    custom_weights[chk.description] = float(wt.value)

        with output:
            output.clear_output()
            try:
                from .. import export, pipeline

                df = session.get("df")
                if df is None:
                    print("Load data first")
                    return

                mode = mode_dd.value
                if mode_dd.value == "manual" and not custom_weights:
                    print("No funds selected")
                    return

                res = pipeline.run_analysis(
                    df,
                    in_start.value,
                    in_end.value,
                    out_start.value,
                    out_end.value,
                    target_vol.value if vol_ck.value else 1.0,
                    0.0,
                    selection_mode=mode,
                    random_n=int(random_n_int.value),
                    custom_weights=custom_weights,
                    rank_kwargs=rank_kwargs,
                    manual_funds=manual_funds,
                    indices_list=list(idx_select.value),
                    benchmarks={b: b for b in bench_select.value},
                )
                if not res:
                    diag = res.diagnostic
                    if diag:
                        print(f"No results ({diag.reason_code}: {diag.message})")
                    else:
                        print("No results")
                else:
                    payload = res.value or {}
                    sheet_formatter = export.make_summary_formatter(
                        payload,
                        in_start.value,
                        in_end.value,
                        out_start.value,
                        out_end.value,
                    )
                    text = export.format_summary_text(
                        payload,
                        in_start.value,
                        in_end.value,
                        out_start.value,
                        out_end.value,
                    )
                    print(text)
                    data = {"summary": pd.DataFrame()}
                    prefix = f"IS_{in_start.value}_OS_{out_start.value}"
                    export.export_data(
                        data,
                        prefix,
                        formats=[out_fmt.value],
                        formatter=cast(Formatter, sheet_formatter),
                    )
            except Exception as exc:
                print("Error:", exc)

    run_btn.on_click(_run_action)

    ui = widgets.VBox(
        [
            step1_box,
            mode_dd,
            random_n_int,
            vol_ck,
            target_vol,
            use_rank_ck,
            next_btn_1,
            rank_box,
            manual_box,
            out_fmt,
            run_btn,
            output,
        ]
    )
    _update_rank_vis()
    _update_inclusion_fields()

    _update_random_vis()

    _update_manual()
    _update_target_vol()

    return ui


#  Once build_ui() is defined, the notebook can do:
#       ui_inputs = build_ui()
#       display(ui_inputs)


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
    "build_ui",
    "canonical_metric_list",
]
