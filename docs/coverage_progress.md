# Coverage Improvement Progress

## Scope
Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

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
  - [x] frequency.py
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

## Current status
- Removed the unavailable `health_summarize` dependency from `requirements.txt` and added an in-repo fallback module so the coverage script can install successfully on Python 3.12.
- Ran targeted coverage with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run --source=trend_analysis.util.frequency -m pytest tests/test_util_frequency_internal.py tests/test_frequency_missing.py tests/test_util_frequency_missing.py`, confirming 100% statement and branch coverage for `trend_analysis.util.frequency` (all 38 tests passed in 7.17s).

## Next steps
- Re-run soft coverage for the broader module set now that the dependency issues are resolved, capturing the lowest-coverage files under the `trend_analysis` namespace.
- Continue lifting coverage file-by-file, starting with the remaining modules listed above that still report <95% coverage or lack tests entirely.
