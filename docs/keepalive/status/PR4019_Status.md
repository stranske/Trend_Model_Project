# Keepalive Status — PR #4019

> **Status:** In progress — reconciled prior test coverage updates and recorded remaining tasks.

## Progress updates
- Round 1: Reviewed recent commits covering rank widgets, walk-forward helpers, and multi-period loaders; updated checklist to reflect those completed tasks.

## Scope
- [ ] Ensure that test coverage exceeds 95% for all program files.

## Tasks
- [ ] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
- [ ] Increase test coverage incrementally for one set of related issues at a time
- [x] ui/rank_widgets.py
- [x] walk_forward.py
- [x] multi_period/loaders.py
- [ ] core/rank_selection.py
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
