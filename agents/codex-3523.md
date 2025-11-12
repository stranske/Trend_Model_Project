<!-- bootstrap for codex on issue #3523 -->

## Scope
- Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered. *(Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest`; exits with one known failure in `tests/test_autofix_pipeline_live_docs.py::test_autofix_pipeline_repairs_live_documents` but produced the coverage report.)*
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py — src/trend_analysis/__init__.py (97%)
  - [x] data.py — src/trend_analysis/data.py (100%)
  - [ ] presets.py — src/trend_analysis/presets.py (60%)
  - [x] harness.py — src/trend_analysis/backtesting/harness.py (100%)
  - [x] regimes.py — src/trend_analysis/regimes.py (99%)
  - [ ] pipeline.py — src/trend_analysis/pipeline.py (57%)
  - [x] validators.py — src/trend_analysis/io/validators.py (99%)
  - [x] run_analysis.py — src/trend_analysis/run_analysis.py (98%)
  - [ ] market_data.py — src/trend_analysis/io/market_data.py (66%)
  - [ ] signal_presets.py — src/trend_analysis/signal_presets.py (75%)
  - [ ] frequency.py — src/trend_analysis/util/frequency.py (79%)
  - [ ] signals.py — src/trend_analysis/signals.py (65%)
  - [x] bootstrap.py — src/trend_analysis/backtesting/bootstrap.py (100%)
  - [ ] risk.py — src/trend_analysis/risk.py (71%)
  - [x] bundle.py — src/trend_analysis/export/bundle.py (99%)
  - [ ] cli.py — src/trend_analysis/cli.py (90%)
  - [ ] optimizer.py — src/trend_analysis/engine/optimizer.py (81%)
  - [x] model.py — src/trend_analysis/config/model.py (99%)
  - [ ] engine.py — src/trend_analysis/multi_period/engine.py (22%)

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage

## Coverage findings
- The soft coverage report highlights the modules above as priority targets. The lowest coverage files are `multi_period/engine.py` (22%), `trend/reporting/unified.py` (7%), `presets.py` (60%), `pipeline.py` (57%), `market_data.py` (66%), and other workflow utilities listed above.
- `run_analysis.py` now reaches 98% coverage after exercising missing CSV configuration checks, nan/missing policy fallbacks, and the detailed output path.

## Next steps
- Add focused unit tests for `trend_analysis.run_analysis.main` covering missing-CSV and nan-policy fallbacks plus the detailed results branch.
- Identify high-leverage seams in `trend_analysis.presets` to exercise the remaining configuration paths (vol-adjust defaults, candidate directory precedence) without heavy filesystem reliance.
