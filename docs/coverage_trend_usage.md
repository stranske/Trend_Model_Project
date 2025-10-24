# Coverage Trend Soft Alerts

This project tracks code coverage trends in CI so coverage regressions are surfaced without blocking merges. The reusable CI workflow parses pytest coverage outputs, appends a summary to the job log, uploads a reusable artifact, and posts a soft warning when coverage drops sharply.

## Workflow behaviour

1. **Coverage parsing** – `tools/coverage_trend.py` reads `coverage.xml` (and falls back to `coverage.json`) to compute the latest line coverage.
2. **Baseline comparison** – The script reads `config/coverage-baseline.json` to determine the expected baseline (`line`) and soft warning drop (`warn_drop`, in percentage points).
3. **Job summary** – Every run appends a "Coverage Trend" section to the GitHub Actions job summary, listing the current coverage, baseline, delta, soft drop limit, and any configured hard minimum.
4. **Artifacts** – The reusable workflow writes `artifacts/coverage-summary.md` (human-readable synopsis) alongside `artifacts/coverage-trend.json` (current coverage, baseline, delta, warn-drop tolerance, and configured hard minimum) and uploads them inside the `gate-coverage-summary` artifact emitted by the primary Python runtime. Gate’s `summary` job republishes the markdown as `gate-coverage-summary.md` and captures normalized stats in `gate-coverage.json` for easy download.
5. **Soft warnings** – On pull requests, if the coverage delta is lower than `-warn_drop`, the workflow posts or updates a 🔶 coverage alert comment (including the current baseline, run coverage, delta, and configured hard minimum) instead of failing CI, and removes the comment once coverage returns within tolerance.

## Updating the baseline

1. Run the reusable workflow (or an equivalent local test run) and ensure the new coverage level is expected.
2. Download the `gate-coverage-summary` (from the Python runtime) or `gate-coverage-summary.md` (from the Gate summary job) artifact from the latest passing run and inspect `coverage-trend.json` to confirm the new percentage.
3. Update `config/coverage-baseline.json` with the new `line` value and, if needed, adjust `warn_drop`.
4. Commit the baseline change with context in the commit message (for example, reference the PR that improved coverage).

## Adjusting soft alert sensitivity

- `warn_drop` defines how many percentage points coverage can fall before the workflow posts a warning comment. Increase it to reduce noise; decrease it to tighten warnings.
- Keep `warn_drop` non-negative. The default of `1.0` means warnings fire when coverage drops more than one point below baseline.

## Local validation

The parsing logic is encapsulated in `tools/coverage_trend.py` and unit-tested via `tests/test_coverage_trend.py`. To run the tests locally:

```bash
pytest tests/test_coverage_trend.py
```

You can also execute the script directly to inspect output files:

```bash
python tools/coverage_trend.py \
  --coverage-xml coverage.xml \
  --baseline config/coverage-baseline.json \
  --artifact-path tmp/coverage-trend.json
```

The command prints nothing by default but will populate `tmp/coverage-trend.json` and append summary information if `--summary-path` (file output) and/or `--job-summary` (append to an existing summary file) are supplied.
