# Trend Model CLI Quickstart

This guide walks through the two console entry points exposed by the Trend Model
package after it has been installed (for example via `pip install -e .`).

## Prerequisites

1. Create and activate a virtual environment (optional but recommended).
2. Install the project in editable mode so the console scripts are available:

   ```bash
   pip install -e .[app]
   ```

   The optional `app` extra pulls in Streamlit so that the `trend-app` command
   can launch the interactive UI.

3. Generate the demo dataset that the sample configuration relies on:

   ```bash
   python scripts/generate_demo.py
   ```

   This writes `demo/demo_returns.csv`, which the sample configuration file
   references.

## Launching the Streamlit UI (`trend-app`)

Run the `trend-app` console script to launch the Streamlit interface bundled in
`streamlit_app/app.py`:

```bash
trend-app
```

The command proxies directly to `streamlit run streamlit_app/app.py`, so any
arguments you provide are forwarded to Streamlit itself. For example, to launch
headless on a specific port:

```bash
trend-app --server.headless true --server.port 8502
```

## Running analyses headlessly (`trend-run`)

The `trend-run` console script executes the full volatility-adjusted trend
pipeline using a YAML or TOML configuration file and produces an HTML report by
default. The repository now ships with a TOML example at `config/trend.toml`
that mirrors the demonstration YAML configuration.

If you pass a Streamlit JSON export instead of YAML/TOML, the `run` command will
auto-detect it and replay the UI settings using the same mapping logic as the
app.

Generate the demo dataset first (see the prerequisites above), then invoke the
command:

```bash
trend-run -c config/trend.toml -o reports/cli_demo.html
```

The example configuration writes the report to the location provided via
`-o/--output`. You can also direct the command to export CSV, JSON, XLSX, or TXT
artefacts by pointing `--artefacts` at a directory and optionally specifying the
formats to emit.

Example:

```bash
trend-run -c config/trend.toml \
  -o reports/cli_demo.html \
  --artefacts reports/artefacts \
  --formats csv json xlsx
```

If your CSV contains fixable date issues (e.g., 11/31/2024), you can opt into
the Streamlit-style correction pass:

```bash
trend-model run \
  -c config/trend.toml \
  -i demo/demo_returns.csv \
  --auto-fix-dates
```

You will be prompted to confirm the corrections. Use `--yes` to skip the
interactive prompt in automation.

### PDF export

Pass `--pdf` to render a PDF alongside the HTML report. This requires the
`fpdf2` dependency (install with `pip install "fpdf2>=2.7"`). When enabled, the
command writes `<output>.pdf` next to the HTML file.

### Configuration tips

* Relative paths inside the configuration are resolved relative to the config
  file, so `config/trend.toml` can reference `demo/demo_returns.csv` without an
  absolute path.
* TOML and YAML configs share the same schema. You can base your own TOML files
  on the provided example or convert existing YAML configs by matching the key
  structure.
* The `seed` parameter ensures deterministic behaviour. Adjust it or pass
  `--seed` on the command line to override per run.

---

For advanced usage—including multi-period backtesting and custom selector or
weighting logic—refer to the broader documentation under `docs/` and the
configuration schema in `config/defaults.yml`.

---

## Replaying Streamlit JSON runs

Use `trend-model run` with the JSON file exported from the Streamlit Model page.
The CLI auto-detects the JSON format and applies the same UI mapping and data
contract checks.

The deprecated `trend-model run-ui` command continues to work but will be
removed in a future release.

## Monte Carlo Commands

Use the Monte Carlo module CLI for the scenario workflow: discover registered
scenarios, validate scenario files, execute simulations, and export charts from
completed bundles. These commands are exposed by `trend_analysis.cli`; run them
with `python -m trend_analysis.cli mc ...` from an installed checkout. Scenario
authoring and output interpretation stay in `docs/phase-3/MonteCarlo.md`.

### List Scenarios (`python -m trend_analysis.cli mc list`)

List registered scenarios from the default registry.

```bash
python -m trend_analysis.cli mc list
```

Filter by tags with `--tags`. The option accepts comma-separated values and can
be repeated. Use `--format json` when another tool should consume the registry
listing; the default format is `table`. Use `--registry PATH` to point at a
custom scenario registry.

```bash
python -m trend_analysis.cli mc list --tags hedge_fund --format json
python -m trend_analysis.cli mc list --tags hedge_fund,example \
  --registry config/scenarios/monte_carlo/index.yml
```

### Validate Scenarios (`python -m trend_analysis.cli mc validate`)

Validate all registered scenarios:

```bash
python -m trend_analysis.cli mc validate
```

Pass a scenario name or a config path to validate a single scenario. Use
`--tags` to validate a subset of registered scenarios and `--registry PATH` to
override the registry location.

```bash
python -m trend_analysis.cli mc validate config/scenarios/monte_carlo/cost_regime_example.yml
python -m trend_analysis.cli mc validate cost_regime_example \
  --registry config/scenarios/monte_carlo/index.yml
```

### Run Scenarios (`python -m trend_analysis.cli mc run`)

Run a scenario by name or config path with `--scenario`, and optionally choose
the output bundle directory with `--out`.

```bash
python -m trend_analysis.cli mc run --scenario cost_regime_example --out outputs/mc_run_1
```

Runtime overrides include `--data` for an alternate CSV/Parquet input,
`--formats` for output formats (`csv`, `json`, `parquet`; comma-separated or
repeatable), `--n-paths`, `--jobs`, `--seed`, `--dry-run`, `--no-progress`, and
`--registry`.

```bash
python -m trend_analysis.cli mc run --scenario cost_regime_example \
  --n-paths 500 --jobs 4 --seed 123
python -m trend_analysis.cli mc run --scenario cost_regime_example \
  --dry-run --n-paths 10
```

### Export Charts (`python -m trend_analysis.cli mc viz`)

Render chart artifacts from an existing Monte Carlo bundle. `--bundle` points
to the bundle directory and `--out` points to the export directory.

```bash
python -m trend_analysis.cli mc viz \
  --bundle outputs/mc_run_1 \
  --out outputs/mc_run_1_exports \
  --charts fan,path_dist,risk_return \
  --html --json --png
```

`--charts` is comma-separated and defaults to `fan,path_dist,risk_return`.
Choose at least one export format with `--html`, `--json`, or `--png`; PNG
export requires a working Kaleido installation.

For scenario authoring and output interpretation, see
`docs/phase-3/MonteCarlo.md`.
