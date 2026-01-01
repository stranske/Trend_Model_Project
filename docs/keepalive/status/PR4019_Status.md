# Keepalive Status — PR #4019

> **Status:** In progress — reconciled prior test coverage updates and recorded remaining tasks.

## Progress updates
- Round 1: Reviewed recent commits covering rank widgets, walk-forward helpers, and multi-period loaders; updated checklist to reflect those completed tasks.
- Round 2: Soft coverage sweep failed during collection due to missing `streamlit`/`fastapi`; captured the lowest-coverage list from `coverage-summary.md` and raised `core/rank_selection.py` to 96% via targeted tests and coverage run.

## Scope
- [ ] Ensure that test coverage exceeds 95% for all program files.

## Tasks
- [ ] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
- [ ] Increase test coverage incrementally for one set of related issues at a time
- [x] ui/rank_widgets.py
- [x] walk_forward.py
- [x] multi_period/loaders.py
- [x] core/rank_selection.py
- [ ] backtesting/harness.py
- [ ] universe.py
- [ ] gui/app.py
- [ ] cli.py
- [ ] data_schema.py (app)
- [ ] multi_period/engine.py
- [ ] diagnostics.py
- [ ] api_server/__init__.py
- [ ] schedules.py
- [ ] pipeline.py
- [ ] time_utils.py
- [ ] api.py
- [ ] app.py
- [ ] data.py
- [ ] health_wrapper.py
- [ ] weight_policy.py

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file

## Soft coverage findings (from `coverage-summary.md`)
Lowest-to-highest coverage entries under 95%:
- 0.0% - trend_analysis/ui/rank_widgets.py
- 79.9% - trend_analysis/core/rank_selection.py
- 84.2% - trend_analysis/walk_forward.py
- 88.9% - trend_analysis/universe.py
- 89.9% - trend_analysis/backtesting/harness.py
- 90.1% - trend_analysis/gui/app.py
- 90.3% - trend_analysis/multi_period/loaders.py
- 90.8% - trend_analysis/diagnostics.py
- 91.4% - trend_portfolio_app/data_schema.py
- 91.6% - trend_analysis/cli.py
- 91.9% - trend_analysis/multi_period/engine.py
- 92.3% - trend_analysis/api_server/__init__.py
- 92.5% - trend_analysis/schedules.py
- 92.6% - trend_analysis/time_utils.py
- 93.0% - trend_analysis/pipeline.py
- 93.6% - trend_portfolio_app/app.py
- 93.7% - trend_analysis/api.py
- 94.7% - trend_analysis/data.py
- 94.7% - trend_portfolio_app/health_wrapper.py
- 94.8% - trend_analysis/portfolio/weight_policy.py
