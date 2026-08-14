# Reproducibility Guide

This guide explains how to ensure reproducible results when running the Trend Model Project, including handling of random seeding and hash randomization.

## Random Seeding

The Trend Model Project uses random numbers in several contexts:
- Portfolio selection (`selection_mode: "random"`)
- Monte Carlo simulations
- Bayesian weighting algorithms

To ensure reproducible results, you can set the `random_seed` parameter in your configuration:

```yaml
portfolio:
  random_seed: 42
```

## Monte Carlo Seeding (Two-Layer Mode)

Monte Carlo scenarios use deterministic path and strategy RNG streams derived from
the scenario seed. The implementation lives in `trend_analysis.monte_carlo.seed`
and follows this scheme:

```python
# Master seed from scenario
master_seed = scenario.monte_carlo.seed

# Per-path seed derivation
def get_path_seed(master_seed: int, path_id: int) -> int:
    return hash((master_seed, path_id)) & 0xFFFFFFFF

# Per-strategy seed derivation (costs, overlays, etc.)
def get_strategy_seed(master_seed: int, path_id: int, strategy_name: str) -> int:
    return hash((master_seed, path_id, strategy_name)) & 0xFFFFFFFF
```

Each path RNG is created with `np.random.default_rng(get_path_seed(...))`. Strategy
RNGs extend the path seed with the strategy name to keep strategy-specific draws
independent while ensuring all strategies see identical simulated paths.

**Note:** The scheme relies on Python's `hash` function. For stable cross-run
results, set `PYTHONHASHSEED` before starting Python (see below).

## Hash Randomization and PYTHONHASHSEED

Python uses hash randomization by default for security reasons, which can affect the ordering of dictionaries and sets. This can impact reproducibility in some scenarios.

### Important Limitation

**Setting `PYTHONHASHSEED` after the Python interpreter has started has no effect.** This environment variable must be set **before** Python starts to control hash randomization.

Some dependencies (including NumPy's test suite) attempt to set `PYTHONHASHSEED` during test execution:

```python
# This is INEFFECTIVE - Python is already running
monkeypatch.setenv('PYTHONHASHSEED', '0')
```

### Proper Hash Seeding Setup

To ensure reproducible hash behavior across runs:

1. **Set `PYTHONHASHSEED` before starting Python:**
   ```bash
   export PYTHONHASHSEED=0
   PYTHONPATH=./src python -m trend.cli run -c config/demo.yml
   ```

2. **Use in test environments:**
   ```bash
   export PYTHONHASHSEED=0
   ./scripts/run_tests.sh
   ```

3. **Add to shell scripts that need reproducibility:**
   ```bash
   #!/bin/bash
   export PYTHONHASHSEED=0
   exec env PYTHONPATH=./src python -m trend.cli run "$@"
   ```

### CI/CD Considerations

For continuous integration environments, ensure hash seeding is set at the environment level:

```yaml
# Example GitHub Actions
env:
  PYTHONHASHSEED: 0
```

```yaml
# Example environment configuration
environment:
  variables:
    PYTHONHASHSEED: "0"
```

## Best Practices for Reproducible Results

1. **Always set random_seed in configuration files**
2. **Set PYTHONHASHSEED=0 before starting Python** (not during runtime)
3. **Document seed values used in published results**
4. **Use version pinning for critical dependencies**
5. **Test reproducibility across different environments**

## Testing Reproducibility

To verify that your setup produces reproducible results:

```bash
# Run 1
export PYTHONHASHSEED=0
PYTHONPATH=./src python -m trend.cli run -c config/demo.yml > results1.txt

# Run 2  
export PYTHONHASHSEED=0
PYTHONPATH=./src python -m trend.cli run -c config/demo.yml > results2.txt

# Compare
diff results1.txt results2.txt
```

Results should be identical when using the same configuration and environment setup.

## Troubleshooting

- **Results vary between runs**: Ensure `PYTHONHASHSEED` is set before Python starts
- **Tests show warnings about hash seeding**: This is expected from NumPy's test suite and can be safely ignored
- **Different results on different systems**: Check Python version, library versions, and floating-point precision

## Deterministic Runs (Issue #723)

The determinism enhancements introduced in Issue #723 standardize how seeds and
hash behavior are applied and provide an optional reproducibility bundle.

### Seed Precedence

When invoking the CLI, the effective random seed is resolved using this order:

1. `--seed <N>` CLI flag (highest precedence)
2. `TREND_SEED` environment variable
3. `config.seed` value inside the loaded YAML config (if present)
4. Fallback default: `42`

### Enforced Hash Seed

The `scripts/trend` wrapper (and Docker image) force `PYTHONHASHSEED=0`
if it is not already set, ensuring stable dict/set iteration ordering. Override
by exporting a different value before calling the script if required.

### Reproducibility Bundle

Add `--bundle` (optionally with a path) to produce a portable archive capturing
the run inputs and metadata:

Contents:
- `run_meta.json`: run_id, config hash, seed, Python + library versions, file hashes
- `metrics.csv|json`: summary metrics
- `summary.txt`: human-readable period summary
- `portfolio.csv` / `benchmark.csv`: time series when available

Example:
```bash
./scripts/trend run -c config/demo.yml --returns demo/demo_returns.csv --seed 777 --bundle outputs/deterministic.zip
unzip -p outputs/deterministic.zip run_meta.json | jq .seed
```

### Verifying Bitwise Stability

```bash
./scripts/trend run -c config/demo.yml --returns demo/demo_returns.csv --seed 111 --bundle first.zip
./scripts/trend run -c config/demo.yml --returns demo/demo_returns.csv --seed 111 --bundle second.zip
diff <(unzip -p first.zip run_meta.json | jq -S .) <(unzip -p second.zip run_meta.json | jq -S .) && echo "Deterministic ✅"
```

If the diff is non-empty, inspect `run_meta.json` differences (look for
unexpected version or ordering changes) and open an issue.

## Content-Addressed Run IDs and Idempotent Reruns

When the CLI is not given an explicit `run_id`, it now derives a
**content-addressed** working `run_id` from the same components used by the
reproducibility bundle: the SHA-256 of the input returns file, the SHA-256 of
the resolved config, and the seed. The shared implementation lives in
`trend_analysis.util.hash.content_run_id` (used by both the bundle and the
working CLI via `working_run_id`), so identical inputs always produce the same
`run_id` and the working id agrees with the bundle id.

The random-UUID fallback is preserved only for the genuinely-unknown case — an
in-memory DataFrame with no source file path (library callers) — so those runs
still receive a unique id.

### Skipping an already-computed run

Because the `run_id` is stable for identical inputs, a prior completed run is
discoverable. `write_run_artifacts` records a stable per-run-id pointer under
`<export.directory>/runs/index/<run_id>.json` alongside the timestamped run
directory. Pass `--skip-if-exists` to short-circuit a rerun when a completed
manifest already exists for the computed `run_id`:

```bash
# First run computes and writes artifacts under demo/exports/runs/...
./scripts/trend run -c config/demo.yml --returns demo/demo_returns.csv

# Second identical run with --skip-if-exists reuses the prior result instead of
# recomputing, printing "already-done: run_id=<id>" and the existing manifest.
./scripts/trend run -c config/demo.yml --returns demo/demo_returns.csv --skip-if-exists
```

Without `--skip-if-exists` the run always recomputes; the flag only changes
whether an existing run is reused, not how the `run_id` is derived.

### Docker Usage

```bash
docker run --rm -e TREND_SEED=202 -v "$PWD/demo/demo_returns.csv":/data/returns.csv -v "$PWD/config/demo.yml":/cfg.yml \
  ghcr.io/stranske/trend-model:latest trend run -c /cfg.yml --returns /data/returns.csv --bundle /tmp/bundle.zip
```

Copy the bundle from the container if needed or mount an output volume.

### Notes

- A parallel `details_sanitized` structure is exposed internally for hashing; the
  original rich objects (DataFrames / Series) remain untouched for tests and
  downstream analysis.
- If you introduce new randomness sources (e.g. custom RNGs, libraries like
  PyTorch), ensure they obey the resolved seed.
