<!-- bootstrap for codex on issue #3470 -->

## Scope
- [ ] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Latest run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest` (1 failure: `tests/test_autofix_pipeline_live_docs.py::test_autofix_pipeline_repairs_live_documents`), followed by `python -m coverage report -m`.
  - Targeted verification for `src/trend_analysis/__init__.py`: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=trend_analysis -m pytest tests/test_trend_analysis_init.py` → `python -m coverage report -m src/trend_analysis/__init__.py` (96% line coverage).
  - Lowest coverage modules (<95%) from the target list, ordered from least to most coverage:
    1. `src/trend_analysis/export/bundle.py` – 7%
    2. `src/trend_analysis/io/validators.py` – 14%
    3. `src/trend_analysis/engine/walkforward.py` – 14%
    4. `src/trend_analysis/util/frequency.py` – 46%
    5. `src/trend_analysis/pipeline.py` – 57%
    6. `src/trend_analysis/io/market_data.py` – 59%
    7. `src/trend_analysis/presets.py` – 60%
    8. `src/trend_analysis/signals.py` – 65%
    9. `src/trend_analysis/run_analysis.py` – 67%
    10. `src/trend_analysis/risk.py` – 71%
    11. `src/trend_analysis/signal_presets.py` – 75%
    12. `src/trend_analysis/engine/optimizer.py` – 81%
    13. `src/trend_analysis/cli.py` – 90%
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py (current: 96%)
  - [x] data.py (current: 98%)
  - [ ] presets.py (current: 60%)
  - [x] harness.py (current: 100%)
  - [x] regimes.py (current: 99%)
  - [ ] pipeline.py (current: 57%)
  - [ ] validators.py (current: 14%)
  - [ ] run_analysis.py (current: 67%)
  - [ ] market_data.py (current: 59%)
  - [ ] signal_presets.py (current: 75%)
  - [ ] frequency.py (current: 46%)
  - [ ] signals.py (current: 65%)
  - [x] bootstrap.py (current: 100%)
  - [ ] risk.py (current: 71%)
  - [ ] bundle.py (current: 7%)
  - [ ] cli.py (current: 90%)
  - [ ] optimizer.py (current: 81%)
  - [x] model.py (current: 99%)
  - [ ] engine.py (current: 14% in `engine/walkforward.py`)

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage
