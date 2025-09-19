# Performance Cache & Incremental Covariance (Phase-2)

This module introduces an in-memory covariance cache (`CovCache`) to remove
repeated O(N^2) covariance recomputation across overlapping multi-period
windows.

## Key Concepts

- Key structure: `(start, end, universe_hash, freq)` gives deterministic retrieval for identical windows/universes.
- Universe hash: stable SHA1 over sorted asset codes, truncated to 8 bytes.
- Payload: `cov`, `mean`, `std`, `n`, `assets` plus (Phase‑2) optional aggregates `s1` (row sums) & `s2` (outer-product sums) enabling O(k^2) rolling updates.
- Incremental path: `incremental_cov_update(payload, old_row, new_row)` updates covariance for a single sliding step without full recompute.
- No eviction by default; LRU + capacity hooks are present for future tuning.

## Configuration Flags

Add these under the new `performance:` section (defaults shown):

```yaml
performance:
  enable_cache: true          # Toggle CovCache usage entirely
  incremental_cov: false      # If true, materialise aggregates and attempt rolling updates
```

Disabling `enable_cache` forces direct covariance recomputation. Setting `incremental_cov: true` adds `s1`/`s2` to each payload and uses them for rolling updates when periods slide with constant window length and identical universe.

## API

```python
from trend_analysis.perf.cache import CovCache, compute_cov_payload

cache = CovCache()
key = cache.make_key("2025-01", "2025-06", ["A","B","C"]) 
payload = cache.get_or_compute(key, lambda: compute_cov_payload(df))
cov = payload.cov  # numpy ndarray
```

### Incremental Algebra

Let `X` be window matrix (rows=time, cols=assets), `n` rows, `m = mean(X)`, `S2 = Σ x_t x_t^T`, `S1 = Σ x_t`.

Covariance identity:
```
cov = (S2 - n * m m^T) / (n - 1)
```
Sliding window (drop `o`, add `n`):
```
S1' = S1 - o + n
S2' = S2 - o o^T + n n^T
m'  = S1'/n
cov' = (S2' - n * m' m'^T) / (n - 1)
```
All operations O(k^2) vs. O(n k^2) full recompute.

### Engine Integration

`multi_period.engine` attaches an experimental `cov_diag` per result when `performance.enable_cache` is true. If `incremental_cov` is enabled and the new window is a one-step slide with identical universe and length, it applies `incremental_cov_update`; otherwise it falls back to full payload construction.

## CLI Observability

The `trend-model run` CLI now surfaces the cache counters at the end of a run whenever a stats payload is present.  Users will see a short block summarising the number of cache entries along with hit/miss and incremental-update totals.  Structured JSONL logging emits a dedicated `{ "event": "cache_stats", ... }` record so automation can track the same values without scraping stdout.

When caching is disabled (e.g. `performance.enable_cache: false`) or no cache snapshot exists in the result payload, the CLI produces **no cache statistics block** and emits **no** `cache_stats` structured event—ensuring true zero overhead in the disabled path.

## Testing

`tests/test_perf_cache.py` verifies:
- Cache hit returns identical object
- Distinct universes create distinct entries

Additional tests:
- `test_incremental_cov.py` validates incremental vs full recompute equivalence.
- `test_cache_disable.py` ensures disabling cache bypasses storage.
- `test_engine_incremental_cov.py` asserts multi-period engine incremental path reproduces full covariance diagonals.

## Roadmap

1. Surface real covariance-dependent metrics through the registry.
2. Add heuristic to detect multi-step jumps and invalidate rolling state.
3. Optional memory cap with eviction metrics reported in run log.
