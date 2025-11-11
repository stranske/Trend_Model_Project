<!-- bootstrap for codex on issue #3470 -->

## Scope
- [ ] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Initial `coverage report` highlighted the lowest-covered targets:
    - `trend_analysis/presets.py` – 15 %【F:coverage_soft_report.md†L3-L3】
    - `trend_analysis/signal_presets.py` – 52 %【F:coverage_soft_report.md†L4-L4】
    - `trend_analysis/util/frequency.py` – 46 %【F:coverage_soft_report.md†L5-L5】
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [ ] __init__.py
  - [ ] data.py
  - [x] presets.py – regression suite lifts coverage to 99 %.【cd1330†L1-L7】
  - [ ] harness.py
  - [ ] regimes.py
  - [ ] pipeline.py
  - [ ] validators.py
  - [ ] run_analysis.py
  - [ ] market_data.py
  - [x] signal_presets.py – dedicated tests bring coverage to 100 %.【cd1330†L1-L7】
  - [x] frequency.py – utility coverage now 100 %.【cd1330†L1-L7】
  - [ ] signals.py
  - [ ] bootstrap.p
  - [ ] risk.py
  - [ ] bundle.py
  - [ ] cli.py
  - [ ] optimizer.py
  - [ ] model.py
  - [ ] engine.py

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage
