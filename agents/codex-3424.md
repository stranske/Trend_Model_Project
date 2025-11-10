<!-- bootstrap for codex on issue #3424 -->

## Scope
Ensure test coverage for Trend Model Project program modules meets or exceeds 95%, with special attention to critical execution paths that currently lack automated verification.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py
  - [x] data.py
  - [x] presets.py
  - [ ] harness.py
  - [ ] regimes.py
  - [ ] pipeline.py
  - [x] validators.py
  - [x] run_analysis.py
  - [ ] market_data.py
  - [ ] signal_presets.py
  - [x] frequency.py
  - [ ] signals.py
  - [ ] bootstrap.py
  - [ ] risk.py
  - [x] bundle.py
  - [ ] cli.py
  - [ ] optimizer.py
  - [ ] model.py
  - [ ] engine.py

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage

## Soft coverage sweep (2025-02-15)
- Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run --source=trend_analysis -m pytest tests/test_validators.py tests/test_io_validators_additional.py tests/test_io_validators_extra.py tests/test_io_validators_negative_paths.py tests/test_io_utils.py test_upload_app.py tests/test_export_bundle.py tests/test_run_analysis_cli_branches.py tests/test_run_analysis_cli_export.py tests/test_default_export.py tests/test_trend_analysis_presets.py tests/test_trend_analysis_presets_additional.py tests/test_trend_analysis_data.py tests/test_trend_analysis_data_additional.py tests/test_trend_analysis_init.py tests/test_trend_analysis_init_extra.py tests/unit/util/test_frequency_comprehensive.py tests/test_frequency_missing.py tests/test_util_frequency_additional.py tests/test_util_frequency_missing.py`
- Result: 259 tests passed in 18.24s with 42 warnings; coverage data stored in `tmp_logs/soft_coverage_report.txt` via `coverage report -m`.

### Lowest coverage modules (<95%)
| Coverage | Module | Missed stmts |
|----------|--------|--------------|
| 0% | `src/trend_analysis/cli.py` | 277 |
| 0% | `src/trend_analysis/gui/app.py` | 340 |
| 0% | `src/trend_analysis/multi_period/engine.py` | 536 |
| 0% | `src/trend_analysis/weights/robust_weighting.py` | 160 |
| 14% | `src/trend_analysis/backtesting/harness.py` | 169 |
| 14% | `src/trend_analysis/engine/walkforward.py` | 130 |
| 15% | `src/trend_analysis/engine/optimizer.py` | 122 |
| 16% | `src/trend_analysis/core/rank_selection.py` | 386 |
| 19% | `src/trend_analysis/regimes.py` | 206 |
| 28% | `src/trend_analysis/logging.py` | 54 |
| 30% | `src/trend_analysis/timefreq.py` | 21 |
| 33% | `src/trend_analysis/perf/rolling_cache.py` | 45 |
| 39% | `src/trend_analysis/pipeline.py` | 346 |
| 44% | `src/trend_analysis/export/__init__.py` | 449 |
| 56% | `src/trend_analysis/risk.py` | 39 |
| 65% | `src/trend_analysis/signals.py` | 17 |
| 66% | `src/trend_analysis/io/market_data.py` | 126 |
| 79% | `src/trend_analysis/api.py` | 13 |

> Modules already exceeding 95% coverage in this run (`src/trend_analysis/__init__.py`, `data.py`, `presets.py`, `io/validators.py`, `run_analysis.py`, `util/frequency.py`, `export/bundle.py`) are reflected as completed subtasks above. The remaining items require additional targeted test development to satisfy the acceptance criteria.
