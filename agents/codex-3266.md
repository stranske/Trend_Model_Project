<!-- bootstrap for codex on issue #3266 -->

## Scope
- Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

## Tasks
- [ ] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
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
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage

## Notes
- Soft coverage run via `./scripts/run_tests.sh` is currently producing 0% results for all modules, suggesting the configuration or instrumentation still needs attention before we can rank files by coverage.
- Targeted coverage sampling with `pytest --cov=trend_analysis.util.frequency tests/test_util_frequency_internal.py -q` confirms existing tests can exercise the frequency helper module with 100% line coverage, but the acceptance criteria remain unmet until the global coverage workflow is reporting accurate figures above 95%.
