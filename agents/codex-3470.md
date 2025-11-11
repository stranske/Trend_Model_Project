<!-- bootstrap for codex on issue #3470 -->

## Scope
- [ ] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Coverage findings (≤95%):
    1. `src/trend/reporting/unified.py` – 88%
    2. `src/trend_analysis/multi_period/engine.py` – 93%
    3. `src/trend/cli.py` – 94%
    4. `src/trend_portfolio_app/app.py` – 94%
    5. `src/trend_analysis/config/legacy.py` – 95%
    6. `src/trend_analysis/export/__init__.py` – 95%
  - Improvements delivered this pass:
    - `src/trend_portfolio_app/monte_carlo/engine.py` – lifted to 100% via new base-class tests.
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
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
  - [x] engine.py

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage
