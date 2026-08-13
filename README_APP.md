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
- `streamlit_app/` multipage Streamlit UI (primary app)
- `src/trend_portfolio_app/` simulation + glue layer
- `tests/` unit tests for schema and policy logic
- `scripts/` convenience launcher
- `examples/legacy_streamlit_app/` archived prototype kept for reference;
  it now contains everything that previously lived under the removed
  `app/streamlit/` tree.

Place the `src/` and `streamlit_app/` folders at the root of your repo (next to your existing `src/trend_analysis`).

## Run the app
```bash
trend app
```

## Browser demo (stlite / Pyodide)

A zero-install browser build of this app lives in [`demo/wasm/`](demo/wasm/). It
runs the real Streamlit UI and the deterministic engine in the browser via
stlite/Pyodide — not Streamlit Community Cloud (per the owner decision in issue
#5343).

```bash
python scripts/build_wasm_demo.py   # (re)generate demo/wasm/manifest.json
```

Runtime modes are selected with the sidebar **Demo mode** switcher or the
`?profile=` URL parameter and enforced by `streamlit_app/demo_profile.py`:

- `presentation_safe` (default): bundled synthetic data only; LLM,
  custom-analysis, and upload surfaces are hidden; no LangChain in the
  requirement set, so a presentation/locked-down PC load has no LLM footprint.
- `public_llm_demo`: exposes the LangChain LLM UI; an explicit CORS-enabled
  OpenAI-compatible endpoint and masked API key are runtime-only and never bundled.

Public demo: <https://stranske.github.io/Trend_Model_Project/>

See [`demo/wasm/README.md`](demo/wasm/README.md) for build/deploy steps and the
live-URL/screenshot/network-evidence verification checklist.

## Secrets (local dev + Streamlit Cloud)

**Local dev (safe)**
  - `.streamlit/secrets.toml`
  - `OPENAI_API_KEY = "..."`

Quick setup script (writes the file and locks permissions):

```
./scripts/setup_streamlit_secrets.sh
```
**Streamlit Cloud (hosted) — synthetic / non-proprietary data only**
  - `OPENAI_API_KEY = "..."`
  - Streamlit Community Cloud is external SaaS. Do **not** use it for real
    proprietary returns. For sharing with synthetic data, use the stlite
    browser demo (issue #5343).

**How Streamlit Cloud differs**
- Local: you control the environment on your machine.
- Cloud: Streamlit runs the app on their infrastructure and loads secrets from
  the app settings (not from repo files).

### Proprietary data: internal / on-prem hosting

To run on **real proprietary returns**, host the app inside your perimeter and
route LLM traffic only through the bundled no-egress proxy. The deterministic
engine and uploaded data never leave the host; LLM calls go through
`trend-llm-proxy` pointed at an authorized no-train upstream (an on-prem Ollama
endpoint keeps prompts in-perimeter too). Set `TREND_LLM_ZONE=disabled` to hide
Streamlit LLM entry points in a zone that has no authorized endpoint while still
running the deterministic engine.

See [`docs/deployment/INTERNAL_HOSTING.md`](docs/deployment/INTERNAL_HOSTING.md)
for the Docker Compose recipe (`docker compose --profile internal up`) and the
operator live-verification checklist.

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

## Confidence interval reporting
The `ci_level` slider controls reporting-only confidence interval annotations.
It does not change portfolio construction, selection, or policy decisions.

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

## Upload safety and caching

- Uploads are restricted to CSV or Excel files up to 10&nbsp;MB. Oversized or
  disallowed uploads are rejected with a descriptive error before validation
  runs.
- Files are written to a dedicated `tmp/uploads/` directory under the repo to
  avoid leaking arbitrary paths.
- The analysis cache is keyed on both the model configuration and the SHA-256
  hash of the uploaded data, ensuring stale results are not reused when either
  inputs or parameters change.

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
