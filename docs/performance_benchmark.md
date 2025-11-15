# Performance Benchmark CI

This workflow (maint-52-perf-benchmark.yml) enforces a regression budget for key
vectorised performance hotspots.

## Monitored Metrics
- `no_cache_mean_s`
- `cache_mean_s`
- `turnover_vectorization.python_mean_s`
- `turnover_vectorization.vectorized_mean_s`
- `turnover_cap_vectorization.largest_gap.python_mean_s`
- `turnover_cap_vectorization.largest_gap.vectorized_mean_s`
- `turnover_cap_vectorization.best_score_delta.python_mean_s`
- `turnover_cap_vectorization.best_score_delta.vectorized_mean_s`

Only raw mean timings are gated (not speedup ratios). A metric fails if its
runtime increases by more than `PERF_REGRESSION_PCT` (default 15%). Equal or
smaller increases pass (non-strict comparator).

## Baseline
Stored at `perf/perf_baseline.json` (generated with rows=1200, cols=35, runs=4).
To regenerate after intentional optimisation:

```bash
python scripts/benchmark_performance.py \
  --rows 1200 --cols 35 --runs 4 --output perf_new.json
mv perf_new.json perf/perf_baseline.json
# Commit with message: chore(perf): update baseline after optimisation (#1158)
```

## Local Comparison
```bash
python scripts/benchmark_performance.py --rows 1200 --cols 35 --runs 4 --output perf_current.json
python scripts/compare_perf.py --current perf_current.json --baseline perf/perf_baseline.json --threshold-pct 15
```

## Fork Safety
Workflow skips on forks via `if: github.repository_owner == 'stranske'`.

## Extending Metrics
Add new measured sections to `benchmark_performance.py` then list raw timing
keys in `MONITORED` inside `scripts/compare_perf.py`. Update baseline and commit.

## Follow-Up (New Issue Suggested)
- Add multi-period scenario timings.
- Record variance over more runs with adaptive thresholds.
