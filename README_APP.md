# Streamlit App + Simulation Engine Starter (ASCII-safe)

This adds a Streamlit front end and thin simulation layer that sits on top of your existing `trend_analysis` package.
It avoids notebooks, uses your pipeline where possible, and isolates policy decisions (hire/fire rules).

## Install (inside your repo's virtualenv)
```bash
pip install streamlit
```
Pydantic is optional; not required for this starter.

## Layout
- `streamlit_app/` multipage Streamlit UI
- `src/trend_portfolio_app/` simulation + glue layer
- `tests/` unit tests for schema and policy logic
- `scripts/` convenience launcher

Place the `src/` and `streamlit_app/` folders at the root of your repo (next to your existing `src/trend_analysis`).

## Run the app
```bash
streamlit run streamlit_app/app.py
```

## Integration with your pipeline
- If available, the code calls `trend_analysis.pipeline.single_period_run(...)` to compute the score frame.
- If import fails, it falls back to a local metrics implementation so the app still runs.

## Monte Carlo
Skeletons for multi-path generation and feature sweeps live under `src/trend_portfolio_app/monte_carlo/`.
