# Streamlit App + Simulation Engine Starter (ASCII-safe)

This adds a Streamlit front end and thin simulation layer that sits on top of your existing `trend_analysis` package.
It avoids notebooks, uses your pipeline where possible, and isolates policy decisions (hire/fire rules).

## Install (inside your repo's virtualenv)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[app]
```
The editable install exposes the `trend` console entry points and pulls in the
Streamlit extras.  The historical `sitecustomize.py` bootstrapper has been
fully removed, so the console scripts are now the *only* supported way to run
the CLI, Streamlit app, demos, and automated tests.

## Layout
- `streamlit_app/` multipage Streamlit UI
- `src/trend_portfolio_app/` simulation + glue layer
- `tests/` unit tests for schema and policy logic
- `scripts/` convenience launcher

Place the `src/` and `streamlit_app/` folders at the root of your repo (next to your existing `src/trend_analysis`).

## Run the app
```bash
trend app
```

The legacy launcher (`scripts/run_streamlit.sh`) still works, but the packaged
command keeps the environment consistent across machines and enforces the
installed-package workflow.

### Unified report downloads

The Results page now exposes "Download report" buttons for HTML and, when the
optional ``fpdf2`` dependency is installed, PDF output. Both use the same
`trend.reporting.generate_unified_report` helper that powers the CLI, ensuring
reports downloaded from the UI are byte-identical to those produced via
``trend report --output``.

## Integration with your pipeline
- If available, the code calls `trend_analysis.pipeline.single_period_run(...)` to compute the score frame.
- If import fails, it falls back to a local metrics implementation so the app still runs.

## Trend presets in the UI and CLI

The Configure page surfaces curated signal presets so users can quickly load
the "Conservative" or "Aggressive" trend settings without tuning every slider.
Selecting a preset updates the trend signal lookback, minimum periods, lag, and
volatility scaling controls alongside the existing portfolio inputs.

The same registry powers the CLI. Run the analysis with a preset by supplying
`--preset`:

```bash
trend-model run --preset conservative -c my_config.yml -i returns.csv
```

Both surfaces share the underlying `TrendSpec` parameters, keeping the Streamlit
app and CLI in sync.

## Monte Carlo
Skeletons for multi-path generation and feature sweeps live under `src/trend_portfolio_app/monte_carlo/`.

## MVP Acceptance (Issue #367)
- Load: CSV with `Date` column; basic validation via data schema. See
  [`docs/validation/market-data-contract.md`](docs/validation/market-data-contract.md)
  for the full ingest contract and metadata propagation rules that the Streamlit
  layer relies on.
- Configure: YAML-like options exposed through UI; choose dates/freq/policy.
- Run: Single and multi-period using existing modules where available.
- View: Metrics tables and key charts (equity, drawdown, weights where applicable).
- View: Results page includes a toggle to overlay a bootstrap 5–95% equity band.
- Export: Zip bundle with returns, events, summary, and a config snapshot.

Matches CLI outputs within normal tolerance; avoids blocking exceptions in demo flow.
