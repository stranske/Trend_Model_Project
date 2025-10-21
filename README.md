# Trend Model Project

[![Gate](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-00-gate.yml/badge.svg?branch=phase-2-dev)](https://github.com/stranske/Trend_Model_Project/actions/workflows/pr-00-gate.yml) [![Codex Verification Guide](https://img.shields.io/badge/codex--verification-docs-blueviolet)](docs/codex-simulation.md)

📚 **CI automation orientation**: Begin with the [Workflow System Overview](docs/ci/WORKFLOW_SYSTEM.md)—it is the canonical first stop for contributors and documents the keep vs retire roster, policy guardrails, observability surfaces, and how Gate, Maint 46, and the agents orchestrator interact. Use the built-in quick orientation checklist, scenario cheat sheet, and bucket reference to map which workflows will fire before you touch YAML. The [Final topology (keep vs retire) table](docs/ci/WORKFLOW_SYSTEM.md#final-topology-keep-vs-retire) enumerates the authoritative roster, and [How to change a workflow safely](docs/ci/WORKFLOW_SYSTEM.md#how-to-change-a-workflow-safely) lays out the guardrails and approval flow. Hop from there to the [workflow catalog](docs/ci/WORKFLOWS.md) when you need trigger, permission, or naming specifics for an individual workflow. Historical notes about retired flows live exclusively in [ARCHIVE_WORKFLOWS.md](docs/archive/ARCHIVE_WORKFLOWS.md).

> **🚀 New User?** → **[Quick Start Guide](docs/quickstart.md)** — Get your first analysis running in under 10 minutes!

This repository contains experiments and utilities for analyzing volatility-adjusted trend portfolios. The Jupyter notebooks demonstrate how to load hedge fund data, apply trend-following rules, and export the results.

For a beginner-friendly overview, see [docs/UserGuide.md](docs/UserGuide.md).

Latest addition: the pipeline now tags market regimes (Risk-On / Risk-Off) via
configurable rolling-return or volatility filters and adds a "Performance by
Regime" table to the CLI, exports, and unified HTML/PDF report. The narrative
summary automatically calls out whichever regime outperforms (or notes when the
strategy behaves similarly across regimes). Tweak lookback, smoothing,
thresholds, and the annualisation flag under the new `regime` section in
`config/defaults.yml` to align the analysis with your preferred market proxy.

📦 **Reusable CI & Automation**: Standardise tests, autofix, and agent automation across repositories using the new reusable workflows documented in [docs/ci_reuse.md](docs/ci_reuse.md). Consumers call `reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`, `reusable-18-autofix.yml`, and the consolidated `agents-70-orchestrator.yml` entry point (which delegates to `reusable-16-agents.yml`).

🧭 **Start here for CI topology**: Read the [Workflow System Overview](docs/ci/WORKFLOW_SYSTEM.md) first, then consult the [Workflow Catalog](docs/ci/WORKFLOWS.md) for the full keep/retire roster, triggers, and permissions.

🗺️ **Workflow system overview**: Start with [docs/ci/WORKFLOW_SYSTEM.md](docs/ci/WORKFLOW_SYSTEM.md) for the high-level buckets, required merge policy, and automation roles.
🧭 **Workflow topology & agent routing**: Learn how workflow buckets, naming, post-CI summaries, and agent labels fit together in [docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md).
🔁 **Layered Test Workflow (Phases 1–3)**: The staged metrics → history/classification → coverage delta reusable workflow implemented in this repository is documented in [docs/ci-workflow.md](docs/ci-workflow.md). All advanced phases are disabled by default for back‑compat.

➡️ **Codex Bootstrap Simulation & Verification Guide:** See [docs/codex-simulation.md](docs/codex-simulation.md) for the hardened workflow design, simulation labels, forced-failure controls, and scenario matrix (T01–T15).

📌 Operational facts for Codex bootstrap (labels, permissions, tokens, PR behavior) are captured in `docs/ops/codex-bootstrap-facts.md`.

## Automated Autofix & Type Hygiene

Pull requests trigger an automated formatting + light type hygiene pass:

- Style: `ruff --fix`, `black`, `isort`, `docformatter`
- Type hygiene: installs missing third‑party stubs (`mypy --install-types`) and runs a narrow allowlist-based script (`scripts/auto_type_hygiene.py`) that appends `# type: ignore[import-untyped]` only for explicitly configured untyped libs (default: `yaml`).
- Idempotent: re-running produces no diffs when clean.
- Safe scope: does **not** mask real type errors or rewrite logic.

If the workflow makes changes it auto-commits a `chore(autofix): ...` patch onto the PR branch. See `docs/autofix_type_hygiene.md` for details, extension steps, and troubleshooting.

Post-CI follow-up continues to run via `maint-46-post-ci.yml`, delegating to the shared reusable autofix composite.



## Notebooks

- `Vol_Adj_Trend_Analysis1.2.TrEx.ipynb` – an earlier version of the analysis.
- `Vol_Adj_Trend_Analysis1.4.TrEx.ipynb` – the current main notebook showing the full workflow.
- Additional historical notebooks can be found under `notebooks/old` and `Old/`.

## Installation

### From PyPI (Recommended)

Install the latest stable release from PyPI:

```bash
pip install trend-model
```

This provides both the legacy ``trend-model`` command and the new unified
``trend`` entrypoint with pipeline, reporting, stress testing, and app launch
modes:

```bash
trend run --help
trend app
```

### Via ``pipx``

For an isolated installation without activating a virtual environment:

```bash
pipx install trend-model
trend app
```

### From Source

For development or to use the latest features:

```bash
git clone https://github.com/stranske/Trend_Model_Project.git
cd Trend_Model_Project
./scripts/setup_env.sh
source .venv/bin/activate
pip install -e .
```

If package installation fails due to network issues, the CLI is still available via:

```bash
# Use the development wrapper script
./scripts/trend-model run --help
./scripts/trend app

# Or run the module directly
PYTHONPATH="./src" python -m trend.cli run --help
```

#### Optional interpreter bootstrap (development only)

The legacy repository-level ``sitecustomize.py`` shim has been replaced with an
opt-in module living under ``trend_model._sitecustomize``.  To restore the
previous behaviour (ensuring ``src/`` is on ``sys.path`` and validating that the
third-party ``joblib`` dependency is used) set the environment variable
``TREND_MODEL_SITE_CUSTOMIZE=1`` **before** launching Python:

```bash
export TREND_MODEL_SITE_CUSTOMIZE=1
python -m trend.cli run --help
```

Without this flag the interpreter starts without any Trend Model-specific side
effects, making it safe to put the repository on ``PYTHONPATH`` during
toolchain bootstrap.

### Docker (Zero-Setup)

For the fastest setup with zero local dependencies:

```bash
# Run the web interface
docker run -p 8501:8501 ghcr.io/stranske/trend-model:latest

# Use the CLI
docker run --rm ghcr.io/stranske/trend-model:latest trend --help
```

Then visit http://localhost:8501 for the interactive web interface.

📖 **See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for complete Docker usage guide.**

## Setup

1. Create a virtual environment and install the required packages:
   ```bash
   ./scripts/setup_env.sh
   ```
 This helper script lives in `scripts/setup_env.sh` and simply wraps
 `python -m venv` followed by `pip install -r requirements.txt`.  It
 creates a `.venv` directory and installs everything from
 `requirements.txt`, including `pandas`, `numpy`, `matplotlib`,
 `ipywidgets`, `PyYAML` and `xlsxwriter`.
2. Launch Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open `Vol_Adj_Trend_Analysis1.4.TrEx.ipynb` and run the cells in
   order.  All notebooks reside at the repository root and depend on the
   `trend_analysis` package for data loading, metrics and exports.

Note: This README now includes a minor formatting tweak used to verify the auto-merge workflow.

## Applying patches

Some updates are provided as patch files. Apply them from the repository root with:

```bash
git apply -p1 <patchfile>
```
The patches usually update modules under the `trend_analysis/` package,
allowing you to rebuild the library incrementally.

Replace `<patchfile>` with the patch you want to apply (for example `codex.patch`).

## Command-line usage

The ``trend`` command is the single entrypoint for the entire toolkit.  Each
subcommand orchestrates a specific workflow:

```bash
# Run the primary analysis pipeline (uses data.csv_path from the config)
trend run --config config/demo.yml

# Generate a structured report bundle in ./perf
trend report --config config/demo.yml --out perf

# Produce a unified HTML report (and optional PDF) alongside artefacts
trend report --config config/demo.yml --output reports --pdf

# Execute a canned stress scenario focused on the 2008 crisis window
trend stress --config config/demo.yml --scenario 2008

# Launch the Streamlit user interface
trend app
```

Provide ``--returns`` to override ``data.csv_path`` from the configuration when
running in ad-hoc mode.  The ``TREND_SEED`` environment variable and ``--seed``
flag follow the same precedence as the legacy CLI.  The ``trend report``
command accepts ``--output`` to write the unified HTML report to a specific
file or directory and ``--pdf`` to request a matching PDF copy (requires the
``fpdf2`` dependency).  These downloads match the Streamlit "Download report"
buttons byte-for-byte so analysts receive identical content regardless of the
interface they use.

The historical ``trend-model`` command remains available for backward
compatibility and forwards to the new implementation.

## Reproducible Results

For consistent, reproducible results across different runs, use the reproducible launcher:

```bash
# Ensures PYTHONHASHSEED is set before Python starts
./scripts/trend-reproducible trend_analysis.run_analysis -c config/demo.yml
```

This is important because setting `PYTHONHASHSEED` after the Python interpreter has started has no effect. The environment variable must be set before Python starts to control hash randomization.

## Structured Run Logging

The CLI and Streamlit UI emit **structured JSONL logs** capturing each major pipeline step with a stable schema. This dramatically simplifies debugging and post‑run analysis.

### CLI

```bash
trend run --config config/demo.yml --returns demo/demo_returns.csv --log-file logs/myrun.jsonl
```

If `--log-file` is omitted a file is created under `outputs/logs/run_<ID>.jsonl` where `<ID>` is a 12‑character run id.

Disable structured logging entirely:

```bash
trend run --config config/demo.yml --returns demo/demo_returns.csv --no-structured-log
```

### Schema (one JSON object per line)

```json
{
   "ts": 1730000000.123,      // POSIX timestamp (float)
   "run_id": "a1b2c3d4e5f6",  // Correlates all lines of a run
   "step": "selection",       // Machine‑friendly step id
   "level": "INFO",           // Logging level
   "msg": "Funds selected",   // Human readable message
   "extra": { "count": 8 }    // Arbitrary JSON‑serialisable context
}
```

### Streamlit UI

After a run completes the **Run Log** pane (Results page) displays:
- Auto‑refreshing tail (last ~500 lines)
- File size + line count
- Error summary table (aggregated by message)
- Download button for the raw JSONL
- Legacy in‑memory event log (will be deprecated once parity established)

### Programmatic Parsing

Use helpers in `trend_analysis.logging`:

```python
from trend_analysis.logging import logfile_to_frame, error_summary
df = logfile_to_frame(Path("outputs/logs/run_<id>.jsonl"))
errs = error_summary(Path("outputs/logs/run_<id>.jsonl"))
```

### Rotation

The handler performs a simple size rotation (default ~1 MB). When the limit is exceeded the active file is renamed with a `.1` suffix and a new file started.

### Typical Steps
`start`, `load_data`, `analysis_start`, `selection`, `weighting`, `benchmarks`, `metrics_build`, `export_start`, `export_complete`, `bundle_complete`, `end`.

### Troubleshooting
- Empty pane: run may not have initialised the structured logger (check `--log-file` path permissions).

## Scalar Metric Memoization (Performance Cache)

Phase‑2 introduces an opt‑in memoization layer for per‑fund scalar metrics (Sharpe, AnnualReturn, etc.) used during ranking and weighting. Enable it via:

```yaml
performance:
   enable_cache: true          # existing covariance cache
   incremental_cov: false
   cache:
      metrics: true             # enable scalar metric series memoization
```

Key facts:
- Zero behavioural change when disabled (default is false if key absent).
- Cache key = `(start, end, ordered_universe, metric_name, stats_cfg_hash)`.
- Safe to clear anytime: `from trend_analysis.core.metric_cache import clear_metric_cache`.
- Introspection: `global_metric_cache.stats()` returns hit/miss counters.

See `docs/metric_cache.md` for full design notes and benchmarking guidance.
- Missing granular steps: selection/weighting keys depend on configuration; ensure the run produced those artifacts.

Structured logging is intentionally additive; if absent the pipeline still produces results normally.

For more details on reproducibility and random seeding, see [docs/ReproducibilityGuide.md](docs/ReproducibilityGuide.md).


## Ranking-based selection

`portfolio.selection_mode` supports a new `rank` value for picking funds by
performance metrics. The defaults for this mode live under `portfolio.rank` in
`config/defaults.yml`. Metrics can be combined using z-scored weights so they
are comparable across scales.
The actual ranking logic is implemented in
`trend_analysis/core/rank_selection.py` and wired into the pipeline via
`trend_analysis/pipeline.py`.

## Information ratio & benchmarks

The pipeline also calculates each portfolio's **information ratio** relative to
one or more benchmarks. The YAML configuration accepts a `benchmarks` mapping
of labels to column names:

```yaml
benchmarks:
  spx: SPX
  tsx: TSX
```

When set, additional `OS IR <label>` columns are appended to the metrics output
and summary Excel sheet. These values measure risk‑adjusted performance versus
the chosen benchmarks.

## Score frame

`trend_analysis.pipeline.single_period_run()` returns a **score frame** – a
DataFrame indexed by fund code with one column per metric listed in
`RiskStatsConfig.metrics_to_run`. The frame also carries `insample_len` and
`period` metadata. `run_analysis()` places this table in the result dictionary
under the key `"score_frame"` so callers can inspect the raw metric values
before any ranking or weighting takes place.

Example:
```python
import pandas as pd
from trend_analysis.pipeline import single_period_run, run_analysis

df = pd.read_csv("my_returns.csv")
sf = single_period_run(df, "2021-01", "2021-03")
print(sf)

# Alternatively, call ``run_analysis`` and grab the same table
res = run_analysis(df, "2021-01", "2021-03", "2021-04", "2021-06", 1.0, 0.0)
score_frame = res["score_frame"]
print(score_frame)
```

## Multi-period demo

Phase 2 introduces a multi-period engine with adaptive weighting. A small
demo dataset and helper script are included to exercise this feature. The
`config/demo.yml` file now enables the rank-based selection mode and adds an
`SPX` benchmark so information ratios and multi-format exports are covered.

Generate the synthetic data and run the demo as follows. The helper script now
cycles through several selector and weighting combinations to cover the main
pipeline features:

```bash
./scripts/setup_env.sh

Troubleshooting: If your VS Code terminal exits with code 1 unexpectedly, it may be because a script was sourced that enabled strict bash options (set -e/pipefail) in your interactive shell. Our setup script is now safe to source, but as a rule prefer executing it: `./scripts/setup_env.sh`. If you ever source a script that sets strict modes, start a new terminal to reset the shell state.
python scripts/generate_demo.py [--no-xlsx]
python scripts/run_multi_demo.py
```
Use `--no-xlsx` to skip the Excel workbook when large files are undesired.

The script prints the number of generated periods and verifies that
weights evolve across them, confirming the Phase 2 logic works end to end.
It now exports each period's score frame alongside the single-period metrics
using ``export.export_data`` so CSV, Excel, JSON **and TXT** files are produced in one call.
The demo also exercises the ``all``, ``random`` and ``manual`` selection modes,
and now calls ``single_period_run`` together with ``calc_portfolio_returns`` to validate
the pipeline helpers. The wrapper ``pipeline.run_analysis`` is also invoked, and
extra calls to ``rank_select_funds`` use the ``percentile`` and ``rank`` transforms so all
scoring options are covered. The demo now checks ``quality_filter`` and the dual calling
patterns of ``select_funds``. It verifies AdaptiveBayesWeighting state persistence and runs the CLI via both the ``-c``
flag and the ``TREND_CFG`` environment variable. Additional checks cover edge
cases of ``AdaptiveBayesWeighting`` such as zero half-life behaviour and
invalid ``prior_mean`` lengths. Finally, the script invokes the
full test suite so every module is covered.
It now also calls ``run_multi_analysis.main`` with a temporary config to verify
the dedicated multi-period CLI works and produces CSV output.
The demo also validates the ``Portfolio`` container by rebalancing
weights from both DataFrame and Series inputs so history tracking remains
tested.

## Demo pipeline (maintenance / CI)

Keep the demo scripts in lock‑step with the exporter and pipeline
functionality. Whenever either changes, run the sequence below and update
`config/demo.yml` or `scripts/run_multi_demo.py` so the demo covers every
code path.

See **[docs/DemoMaintenance.md](docs/DemoMaintenance.md)** for a concise
checklist of these steps.

1. **Bootstrap the environment**
   ```bash
   ./scripts/setup_env.sh
   ```
2. **Generate the demo dataset**
   ```bash
   python scripts/generate_demo.py [--no-xlsx]
   ```
   Pass `--no-xlsx` to avoid creating the Excel copy.
3. **Run the full demo pipeline and export checks**
   ```bash
   python scripts/run_multi_demo.py
   ```
4. **Run the test suite**
   ```bash
   ./scripts/run_tests.sh
   ```

## Interactive GUI

An interactive configuration GUI is bundled with the project. Launch it within Jupyter by running:

```python
from trend_analysis.gui import launch
launch()
```

The widget reloads the last saved settings from ~/.trend_gui_state.yml and persists new changes after a successful run. Installing the optional ipydatagrid package unlocks spreadsheet-style YAML editing; otherwise the GUI falls back to simpler controls.


## Streamlit App (MVP)

An interactive Streamlit app lets you upload data, configure a simulation, run, and export results.

- Entry point: `streamlit_app/app.py`
- Pages: Upload → Configure → Run → Results → Export

Run it locally:

```bash
./scripts/setup_env.sh
./scripts/run_streamlit.sh
```

Tips
- You can load the included demo dataset from the Upload page.
- The Upload page accepts CSV and Excel files; it validates a Date column and monthly frequency.
- The Results page shows equity and drawdown charts and lets you export a bundle (ZIP) of key outputs.


## Releases

This project uses automated releases to PyPI with semantic versioning.

### For Users
- Install latest stable version: `pip install trend-analysis`
- View releases: [GitHub Releases](https://github.com/stranske/Trend_Model_Project/releases)
- Package on PyPI: [trend-analysis](https://pypi.org/project/trend-analysis/)

### For Maintainers
Create releases by pushing version tags:
```bash
git tag v0.1.0
git push origin v0.1.0
```

The automated workflow will:
- Build distribution packages (wheel + source)
- Test installation 
- Generate changelog from conventional commits
- Publish to PyPI
- Create GitHub release with changelog

For detailed release process documentation, see [docs/release-process.md](docs/release-process.md).

## Testing

Install the project dependencies (such as `pandas`, `numpy` and `PyYAML`) before running the test suite. This can be done using the setup script, which
creates a virtual environment and installs everything from `requirements.txt`
(including `pytest`):

```bash
./scripts/setup_env.sh
```
The script in `scripts/setup_env.sh` prepares a throw-away environment
and installs test dependencies such as `pytest`.

or by installing directly with pip (Python 3.11+ recommended):

```bash
pip install -r requirements.txt pytest coverage
```
`requirements.txt` lives at the repository root and lists every
dependency required to run the analysis. `pytest` and `coverage`
are additional development dependencies needed for running the test
suite.

Once the dependencies are installed, run the tests with coverage enabled:

```bash
coverage run --rcfile .coveragerc.core -m pytest --maxfail=1 --disable-warnings
coverage report -m
```
All unit tests reside in the `tests/` directory and enforce 100 % branch
coverage.

Alternatively, after installing the dependencies, you can use the
helper script to execute the test suite in one step. The script accepts a
`COVERAGE_PROFILE` environment variable (`core` by default) to switch between
coverage configurations:

```bash
# Run tests with the default core profile
./scripts/run_tests.sh

# Run tests with the full profile
COVERAGE_PROFILE=full ./scripts/run_tests.sh
```
This convenience wrapper (under `scripts/run_tests.sh`) assumes
`pytest` and `coverage` are available in the environment and then runs the
same coverage command as above.

## Deterministic Runs & Reproducibility Bundles

Issue #723 introduced first‑class deterministic execution and portable run
bundles. The pipeline now enforces a stable hash seed and exposes explicit
controls for random seeds and artifact capture.

Key points:
- Hash randomisation: `PYTHONHASHSEED` is forced to `0` by the wrapper script
   (`scripts/trend-model`) and Docker image unless you override it. This makes
   any ordering that accidentally depends on dict/set hashing stable across
   processes.
- Seed precedence (highest wins): CLI `--seed` flag > `TREND_SEED` env var >
   `config.seed` (if present) > default `42`.
- The CLI attaches a combined portfolio return series plus (first) benchmark
   so downstream reproducibility tooling can compute digests deterministically.

### Basic deterministic invocation

```bash
./scripts/trend-model run -c config/demo.yml -i demo/demo_returns.csv --seed 123
```

Repeat runs with the same inputs + seed produce identical `run_id` values in
the bundle manifest (see below) and identical metrics output.

### Reproducibility bundle

Add the `--bundle` flag to write a compressed archive containing:
- Run manifest (`run_meta.json`) with: `run_id`, config hash, seed, package
   versions, file hashes
- Metrics table (`metrics.csv` / JSON)
- Summary text file
- Portfolio and benchmark time series (when available)

Usage examples:
```bash
# Default bundle name (analysis_bundle.zip) in CWD
./scripts/trend-model run -c config/demo.yml -i demo/demo_returns.csv --bundle

# Custom bundle path (directory or file). If a directory is supplied the
# file name defaults to analysis_bundle.zip inside it.
./scripts/trend-model run -c config/demo.yml -i demo/demo_returns.csv --bundle outputs/my_run.zip
```

Inspect the manifest quickly:
```bash
unzip -p analysis_bundle.zip run_meta.json | jq .
```

### Overriding seeds via environment

```bash
TREND_SEED=999 ./scripts/trend-model run -c config/demo.yml -i demo/demo_returns.csv
```

This is equivalent to `--seed 999` unless the CLI flag is also provided, in
which case the flag wins.

### Docker deterministic run

```bash
docker run --rm -e PYTHONHASHSEED=0 -e TREND_SEED=123 \
   -v "$PWD/demo/demo_returns.csv":/data/returns.csv \
   -v "$PWD/config/demo.yml":/cfg.yml \
   ghcr.io/stranske/trend-model:latest trend-model run -c /cfg.yml -i /data/returns.csv --bundle
```

All randomness (Python + NumPy) is seeded; if you introduce additional RNG
sources (e.g. `random.Random`, `torch`, `scipy` stochastic routines) ensure
they are also seeded inside the pipeline.

### Verifying determinism locally

```bash
./scripts/trend-model run -c config/demo.yml -i demo/demo_returns.csv --seed 321 --bundle
unzip -p analysis_bundle.zip run_meta.json | jq -r .run_id > first.id
./scripts/trend-model run -c config/demo.yml -i demo/demo_returns.csv --seed 321 --bundle my2.zip
unzip -p my2.zip run_meta.json | jq -r .run_id > second.id
diff -u first.id second.id && echo "Deterministic ✅"
```

If the diff command produces output, open an issue with the differing
`run_meta.json` files so the source of non‑determinism can be traced.

## Contributor Style & Lint Workflow

To avoid CI style job failures:

1. Run the fast development validations during active editing:
   ```bash
   ./scripts/dev_check.sh --changed --fix
   ./scripts/validate_fast.sh --fix
   ```
2. Before pushing a branch, mirror the CI style job exactly (pinned Black/Ruff/Mypy versions used by the unified workflow):
   ```bash
   ./scripts/style_gate_local.sh
   ```
   If it reports failures, fix them with:
   ```bash
   black .
   ruff check --fix .
   # address any mypy errors reported by the script
   ```
3. Only then run the comprehensive branch checks (optional for large changes):
   ```bash
   ./scripts/check_branch.sh --fast --fix
   ```

The script `scripts/style_gate_local.sh` sources version pins from `.github/workflows/autofix-versions.env` ensuring local checks match CI. Add it to your pre-push routine (or a custom git hook) to eliminate format drift caused by differing global tool versions.

## Contributing & Quality Gate

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow.
Common local checks before pushing:
```bash
./scripts/quality_gate.sh        # style + adaptive validation
./scripts/quality_gate.sh --full # adds comprehensive branch check
```
Install the optional pre-push hook:
```bash
./scripts/install_pre_push_style_gate.sh
```
This prevents accidental pushes that fail the CI style checks.

Gate (our required PR workflow) now detects documentation-only diffs internally. When a pull request only touches Markdown, `docs/`, or `assets/` paths, the heavy Python and Docker jobs are skipped while Gate records the standard "docs-only" notice in the job summary and reports success within seconds. Legacy runs also clean up the deprecated docs-only PR comment so conversations stay noise-free. All other changes continue to fan out to the full test matrix.

