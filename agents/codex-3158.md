# Codex Bootstrap Checklist for Issue #3158

## Scope
Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
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
  - [ ] engine.py

## Acceptance Criteria
- [ ] Test coverage exceeds 95% for each file.
- [ ] Essential functions for the program have full test coverage.

## Progress Notes

- **2025-10-30:** Ran `coverage run` with the `core` profile to capture current coverage. The suite aborted early because `tests/test_selector_weighting.py::test_selector_weighting_autofix_diagnostics` fails when `auto_type_hygiene` introduces a `# type: ignore` comment. Coverage data still captured the following baseline (all registering 0% coverage and therefore top priority):
  - `src/trend_analysis/__init__.py`
  - `src/trend_analysis/data.py`
  - `src/trend_analysis/presets.py`
  - `src/trend_analysis/backtesting/harness.py`
  - `src/trend_analysis/regimes.py`
  - `src/trend_analysis/pipeline.py`
  - `src/trend_analysis/io/validators.py`
  - `src/trend_analysis/run_analysis.py`
  - `src/trend_analysis/io/market_data.py`
  - `src/trend_analysis/signal_presets.py`
  - `src/trend_analysis/util/frequency.py`
  - `src/trend_analysis/signals.py`
  - `src/trend_analysis/backtesting/bootstrap.py`
  - `src/trend_analysis/risk.py`
  - `src/trend_analysis/export/bundle.py`
  - `src/trend_analysis/cli.py`
  - `src/trend_analysis/engine/optimizer.py`
  - `src/trend_analysis/multi_period/engine.py`
- The failing integration test needs to be addressed (or temporarily skipped) before tightening coverage gates; until that is resolved we should expect subsequent coverage runs to exit with a failure code.
- **2025-10-30:** Added in-repo typing stubs for `yaml` under `src/trend_analysis/stubs/` and pointed mypy to the directory via `mypy_path`. This keeps the `test_selector_weighting_autofix_diagnostics` integration run from failing when `yaml` stubs are absent. Also introduced focused unit tests for `trend_analysis.__init__` that exercise the lazy import machinery and metadata fallback so the module can accumulate coverage once the full suite runs under coverage.
