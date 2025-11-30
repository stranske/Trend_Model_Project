<!-- bootstrap for codex on issue #3384 -->

## Scope
- [ ] Ensure per-file test coverage reaches at least 95% across the program codebase.
- [ ] Backfill tests for essential functionality that currently lacks coverage while excluding workflow pipeline and non-Python assets.

## Tasks
- [ ] Run the soft coverage report and list files with coverage below 95% or lacking critical tests, ordered from lowest coverage upward.
- [ ] Increase coverage incrementally for one related area or file at a time:
  - [ ] __init__.py
  - [ ] data.py
  - [ ] presets.py
  - [ ] harness.py
  - [ ] regimes.py
  - [ ] pipeline.py
  - [ ] validators.py
  - [ ] run_analysis.py
  - [ ] market_data.py
  - [ ] signal_presets.py
  - [ ] frequency.py
  - [ ] signals.py
  - [ ] bootstrap.p
  - [ ] risk.py
  - [ ] bundle.py
  - [ ] cli.py
  - [ ] optimizer.py
  - [ ] model.py
  - [ ] engine.py

## Acceptance criteria
- [ ] Every targeted file listed above reports at least 95% coverage in the latest test run.
- [ ] All essential program functions have dedicated test coverage.

## Progress log
- 2024-05-19: Initial status review; no scope items, tasks, or acceptance criteria have been completed yet.
