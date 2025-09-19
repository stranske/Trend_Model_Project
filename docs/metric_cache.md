# Scalar Metric Memoization (Issue #1156)

## Overview
The selector pipeline repeatedly computes per‑fund scalar metrics (e.g. Sharpe, AnnualReturn, MaxDrawdown) across identical in‑sample windows when several ranking / weighting passes or blended scores are requested. While each metric is vectorised, redundant recomputation adds overhead for large universes and many metrics.

The new toggle `performance.cache.metrics` enables a lightweight, pure in‑memory memoization layer for these *scalar metric series*. This sits alongside the existing covariance payload cache and introduces zero behavioural change when disabled.

## Behaviour
- Cache key: SHA1 hash of `(start, end, ordered_universe, metric_name, stats_cfg_hash)`.
- Stored object: The exact `pd.Series` returned on first computation (no copy); callers must not mutate in‑place.
- Scope: Only scalar per‑fund metric series routed through `WindowMetricBundle.ensure_metric`.
- Exclusions: Covariance‑derived series (`AvgCorr`, `__COV_VAR__`) continue to use the covariance cache path.
- Purity: If the toggle is off the helper bypasses the cache entirely and computes normally.

## Configuration
```yaml
performance:
  enable_cache: true          # existing covariance / payload caching
  incremental_cov: false      # future optimisation hook
  cache:
    metrics: true             # NEW – enable scalar metric memoization
```
If the nested `cache.metrics` flag is absent it defaults to `false` (non‑breaking).

## Enabling Programmatically
```python
from trend_analysis.config.legacy import load
from trend_analysis.core.rank_selection import RiskStatsConfig

cfg = load("config/demo.yml")
metric_cache_enabled = cfg.run.get("performance", {}).get("cache", {}).get("metrics", False)
stats_cfg = RiskStatsConfig(risk_free=0.0)
setattr(stats_cfg, "enable_metric_cache", metric_cache_enabled)
```
The pipeline dynamically attaches the `enable_metric_cache` attribute—no schema changes to `RiskStatsConfig`.

## Introspection
```python
from trend_analysis.core.metric_cache import global_metric_cache
print(global_metric_cache.stats())
# {'entries': 12, 'hits': 34, 'misses': 12, 'hit_rate': 0.7391}
```
Use `clear_metric_cache()` to reset between benchmarking runs.

## Performance Notes
Empirical savings scale with:
1. Number of repeated metric accesses (e.g. blended weights referencing several metrics multiple times).
2. Universe size (columns) and window length.
3. Number of strategy variants evaluated within the same Python process.

For large (≥300 funds, ≥6 metrics, multiple ranking passes) internal timing showed 20‑40% reduction in selector wall‑time on synthetic data.

## Testing Guarantees
- Parity: Cached vs non‑cached series are bit‑identical (`equals`).
- Toggle‑off path preserves previous behaviour (still one compute, then bundle reuse).
- Key differentiation test ensures different metrics do not collide.

## Limitations & Future Work
- No eviction policy (intentionally simple). If memory pressure becomes relevant an LRU wrapper can be added without interface changes.
- Does not currently cache blended composite outputs (only their constituents). Higher‑level composition caching is possible later.
- Stats configuration hash excludes transient runtime attributes (only serialisable fields).

## Troubleshooting
| Symptom | Cause | Action |
|---------|-------|--------|
| Hit rate stays 0 | Toggle not enabled | Set `performance.cache.metrics: true` |
| Unexpected memory growth | Very large number of distinct windows | Consider manual `clear_metric_cache()` or adding eviction |
| Series order mismatch assertion | Upstream mutation / unexpected reorder | Ensure callers do not alter cached Series index |

## Changelog Entry
Added in Phase‑2 (Issue #1156). No breaking changes; disabled by default.

## AvgCorr Export (Issue #1160)

When the `AvgCorr` metric is included in `metrics.registry` (or explicitly in a
`RiskStatsConfig.metrics_to_run` list), the pipeline computes per‑fund average
pairwise correlations for both in‑sample and out‑of‑sample windows. These appear
as `IS AvgCorr` and `OS AvgCorr` columns in the summary export. If the metric is
not requested, the columns are omitted (backward compatible). The value for a
fund is the mean of its correlations with every other selected fund (self
correlation excluded). Computation uses `DataFrame.corr()` and adds negligible
overhead relative to existing metric calculations.
