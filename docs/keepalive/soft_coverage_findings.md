# Soft coverage findings

Results from the latest soft coverage sweep attempts on 2025-01-20.

## Execution notes

- Full-suite coverage (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest`) failed during collection because optional dependencies (`streamlit`, `fastapi`) are not installed.
- A full-suite rerun with those tests ignored ran longer than the command timeout.
- The ranked list below is derived from `coverage-summary.md` (last updated 2025-12-04) until a full-suite soft coverage run can complete in this environment.

## Files below 95% coverage (lowest first)

| Coverage | File |
|----------|------|
| 0.0% | `src/trend_analysis/ui/rank_widgets.py` |
| 79.9% | `src/trend_analysis/core/rank_selection.py` |
| 84.2% | `src/trend_analysis/walk_forward.py` |
| 88.9% | `src/trend_analysis/universe.py` |
| 89.9% | `src/trend_analysis/backtesting/harness.py` |
| 90.1% | `src/trend_analysis/gui/app.py` |
| 90.3% | `src/trend_analysis/multi_period/loaders.py` |
| 90.8% | `src/trend_analysis/diagnostics.py` |
| 91.4% | `src/trend_portfolio_app/data_schema.py` |
| 91.6% | `src/trend_analysis/cli.py` |
| 91.9% | `src/trend_analysis/multi_period/engine.py` |
| 92.3% | `src/trend_analysis/api_server/__init__.py` |
| 92.5% | `src/trend_analysis/schedules.py` |
| 92.6% | `src/trend_analysis/time_utils.py` |
| 93.0% | `src/trend_analysis/pipeline.py` |
| 93.6% | `src/trend_portfolio_app/app.py` |
| 93.7% | `src/trend_analysis/api.py` |
| 94.7% | `src/trend_analysis/data.py` |
| 94.7% | `src/trend_portfolio_app/health_wrapper.py` |
| 94.8% | `src/trend_analysis/portfolio/weight_policy.py` |
