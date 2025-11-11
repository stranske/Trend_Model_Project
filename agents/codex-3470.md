<!-- bootstrap for codex on issue #3470 -->

## Scope
- [ ] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run -m pytest` followed by `python -m coverage report -m`.
  - Lowest-covered modules (baseline snapshot):
    1. `src/trend_analysis/export/bundle.py` – 7 %
    2. `src/trend_analysis/run_analysis.py` – 13 %
    3. `src/trend_analysis/io/validators.py` – 14 %
    4. `src/trend_analysis/multi_period/engine.py` – 18 %
    5. `src/trend_analysis/engine/optimizer.py` – 23 %
    6. `src/trend_analysis/data.py` – 30 %
    7. `src/trend_analysis/config/model.py` – 44 %
    8. `src/trend_analysis/util/frequency.py` – 46 %
    9. `src/trend_analysis/io/market_data.py` – 54 %
    10. `src/trend_analysis/pipeline.py` – 57 %
    11. `src/trend_analysis/presets.py` – 60 %
    12. `src/trend_analysis/signals.py` – 65 %
    13. `src/trend_analysis/__init__.py` – 68 %
    14. `src/trend_analysis/risk.py` – 71 %
    15. `src/trend_analysis/signal_presets.py` – 75 %
    16. `src/trend_analysis/cli.py` – 87 %
    17. `src/trend_analysis/regimes.py` – 99 %
    18. `src/trend_analysis/backtesting/harness.py` – 100 %
    19. `src/trend_analysis/backtesting/bootstrap.py` – 100 %
- [ ] Increase test coverage incrementally for one set of related issues or one file below at a time.
  - [ ] __init__.py
  - [ ] data.py
  - [ ] presets.py
  - [ ] harness.py
  - [ ] regimes.py
  - [ ] pipeline.py
  - [ ] validators.py
  - [x] run_analysis.py – Added `tests/test_run_analysis.py` to exercise CLI argument resolution, legacy loader compatibility, and export defaults. Targeted coverage run reports 98 % coverage.
  - [ ] market_data.py
  - [ ] signal_presets.py
  - [ ] frequency.py
  - [ ] signals.py
  - [ ] bootstrap.py
  - [ ] risk.py
  - [ ] bundle.py
  - [ ] cli.py
  - [ ] optimizer.py
  - [ ] model.py
  - [ ] engine.py

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file listed above, based on focused coverage runs for each module.
- [ ] Essential functions for the program have full test coverage through the targeted suites listed above.
