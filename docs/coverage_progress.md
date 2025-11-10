# Coverage Improvement Progress

## Scope
Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py
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
  - [ ] bootstrap.py
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
- Captured a baseline “soft coverage” snapshot by running `python -m coverage run -m pytest` at the repository root followed by `python -m coverage report`, which highlighted that overall coverage sits at 32% with a large set of core modules below the 95% target.【adabde†L1-L4】【86999e†L1-L59】
- Added a dedicated regression suite in `tests/trend_analysis/test_package_init.py` that exercises the dataclass guard rails, module registration proxy, lazy loader, conditional export wiring, and version metadata fallbacks in `trend_analysis/__init__.py`, lifting the package initializer to 96% statement coverage (10 tests, 8.66s).【6d3530†L1-L2】【f92315†L1-L5】
- Verified that the new coverage data shows only branch partials remaining for optional import paths, with all executable statements under `trend_analysis/__init__.py` covered by the fresh tests.【f92315†L1-L5】

### Lowest coverage modules (sorted ascending; <95%)

| File | Coverage |
| --- | ---: |
| `src/trend/reporting/unified.py` | 7% |
| `src/trend_analysis/api.py` | 23% |
| `src/trend_analysis/data.py` | 29% |
| `src/trend_analysis/presets.py` | 15% |
| `src/trend_analysis/regimes.py` | 11% |
| `src/trend_analysis/pipeline.py` | 38% |
| `src/trend_analysis/validators.py` | 14% |
| `src/trend_analysis/run_analysis.py` | 13% |
| `src/trend_analysis/market_data.py` | 54% |
| `src/trend_analysis/signal_presets.py` | 52% |
| `src/trend_analysis/util/frequency.py` | 46% |
| `src/trend_analysis/signals.py` | 65% |
| `src/trend_analysis/risk.py` | 70% |
| `src/trend_analysis/export/bundle.py` | 7% |
| `src/trend_analysis/cli.py` | 10% |
| `src/trend_analysis/engine/optimizer.py` | 22% |
| `src/trend_analysis/config/model.py` | 44% |
| `src/trend_analysis/multi_period/engine.py` | 18% |

> **Note:** Auxiliary probe/automation helpers remain at 0% because they are excluded from current testing scope; focus remains on the core orchestration modules enumerated above.【86999e†L1-L59】

## Next steps
- Expand the new initializer regression suite to cover the remaining partial branches (lazy import cache re-population and metadata success path) if business cases demand it.
- Prioritise targeted suites for the lowest coverage modules called out above, beginning with `trend_analysis/api.py`, `trend_analysis/pipeline.py`, and `trend_analysis/run_analysis.py`, to drive each file beyond the 95% threshold while exercising essential functionality end-to-end.
