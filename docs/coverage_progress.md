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
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage

## Current status
- Attempted to execute `./scripts/run_tests.sh` to generate a coverage report, but the command failed during dependency installation because the `health_summarize` package is unavailable for Python 3.12. This prevents obtaining the coverage metrics required for the first task.

## Next steps
- Determine whether an alternative package or version is available for `health_summarize`, or adjust the test runner configuration so coverage can complete without it.
- Once dependencies install successfully, rerun the coverage command and capture the coverage gaps per file.
