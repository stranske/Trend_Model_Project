# Performance benchmark workflow

The repository ships with an automated benchmark that exercises our
vectorisation hotspots and enforces a basic regression budget in CI.

## What the workflow runs

The workflow defined in `.github/workflows/perf-benchmark.yml` executes on
push, pull requests, and a nightly schedule (05:00 UTC).  It runs only on the
canonical repository (`stranske/Trend_Model_Project`) to avoid burning cycles
on forks.

Each run performs the following steps:

1. Install the project in editable mode with runtime dependencies.
2. Execute `scripts/benchmark_performance.py` using a deterministic dataset
   (`--rows 600 --cols 35 --runs 3`).  The script emits `benchmark_perf.json`
   with aggregate timing and speedup information.
3. Compare the new results against `perf_baseline.json` via
   `scripts/check_perf_regression.py`.  If any runtime slows down or speedup
   drops by more than the allowed threshold, the job fails.
4. Upload the generated JSON as an artifact so the raw numbers remain auditable.

The default regression budget is 15%.  Set the environment variable
`PERF_REGRESSION_PCT` (repository or organisation variable) to tighten or
loosen this gate.  Values greater than 1 are treated as percentages (e.g. `15`
means 15%), while fractional values (e.g. `0.1`) are interpreted directly.

## Updating the baseline

When intentional performance changes occur, regenerate the baseline and commit
the updated file alongside your code changes:

```bash
python scripts/benchmark_performance.py \
  --rows 600 --cols 35 --runs 3 --output perf_baseline.json
```

Always inspect the diff to make sure the new numbers look reasonable.  If the
changes are expected to make parts of the benchmark slower, document the reason
in your pull request so reviewers understand why the baseline shifted.

