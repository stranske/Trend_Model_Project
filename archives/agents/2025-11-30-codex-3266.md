<!-- bootstrap for codex on issue #3266 -->

## Scope
- Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

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
  - [x] run_analysis.py
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
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage

## Notes
- Soft coverage snapshot with `pytest --cov=trend_analysis.run_analysis --cov-report=term-missing tests -k run_analysis -q` highlighted `src/trend_analysis/run_analysis.py` as the next priority at 96% coverage (lines 53–61 were still uncovered).
- Added `test_main_uses_nan_fallback_and_default_exports` to `tests/test_run_analysis_additional.py` to exercise the legacy fallback behaviour for `nan_*` parameters and the default export configuration path. The run now records 98% line coverage for `run_analysis.py`, with only the unreachable `nan_limit`/`nan_policy` absence branch remaining.
- Retained the earlier targeted sampling command (`pytest --cov=trend_analysis.util.frequency tests/test_util_frequency_internal.py -q`) to confirm that existing coverage for `util.frequency` stays at 100%, ensuring regressions are caught while broader instrumentation issues are investigated.
