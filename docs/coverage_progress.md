# Coverage Improvement Progress

## Scope
Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

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
  - [x] run_analysis.py
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

## Latest soft coverage sweep
- Executed `python -m coverage run --source=src -m pytest` to take a baseline snapshot (terminated after 301 tests when the run exceeded the soft time budget).【54fb89†L1-L45】
- Captured the resulting module-level percentages with `python -m coverage report --skip-covered --skip-empty > coverage_report.txt` to drive the triage list.【b1981e†L1-L64】

### Files under 95% coverage (ascending)
| File | Coverage |
| --- | ---: |
| `src/health_summarize/__init__.py` | 0.0% |
| `src/trend_analysis/_autofix_probe.py` | 0.0% |
| `src/trend_analysis/proxy/__main__.py` | 0.0% |
| `src/trend_portfolio_app/__main__.py` | 0.0% |
| `src/trend_portfolio_app/health_wrapper.py` | 0.0% |
| `src/trend_portfolio_app/health_wrapper_runner.py` | 0.0% |
| `src/trend/reporting/unified.py` | 7.0% |
| `src/trend_analysis/export/__init__.py` | 7.0% |
| `src/trend_analysis/export/bundle.py` | 7.0% |
| `src/trend_analysis/multi_period/replacer.py` | 8.0% |
| `src/trend_analysis/cli.py` | 10.0% |
| `src/trend_analysis/weights/robust_weighting.py` | 12.0% |
| `src/trend_analysis/metrics/rolling.py` | 13.0% |
| `src/trend_analysis/run_analysis.py` | 13.0% |
| `src/trend_analysis/weights/equal_risk_contribution.py` | 13.0% |
| `src/trend_analysis/weights/hierarchical_risk_parity.py` | 13.0% |
| `src/trend/cli.py` | 14.0% |
| `src/trend_analysis/engine/walkforward.py` | 14.0% |
| `src/trend_analysis/io/validators.py` | 14.0% |
| `src/trend_analysis/presets.py` | 15.0% |
| `src/trend_analysis/core/rank_selection.py` | 17.0% |
| `src/trend_analysis/rebalancing/strategies.py` | 17.0% |
| `src/trend_analysis/viz/charts.py` | 17.0% |
| `src/trend_analysis/io/utils.py` | 18.0% |
| `src/trend_analysis/multi_period/engine.py` | 18.0% |
| `src/trend_analysis/regimes.py` | 20.0% |
| `src/trend_portfolio_app/io_utils.py` | 22.0% |
| `src/trend_analysis/config/bridge.py` | 23.0% |
| `src/trend_analysis/engine/optimizer.py` | 23.0% |
| `src/trend_model/cli.py` | 24.0% |
| `src/trend_analysis/metrics/attribution.py` | 26.0% |
| `src/trend_analysis/util/hash.py` | 26.0% |
| `src/trend_analysis/proxy/cli.py` | 27.0% |
| `src/trend_analysis/logging.py` | 28.0% |
| `src/trend_analysis/run_multi_analysis.py` | 28.0% |
| `src/trend_analysis/weights/risk_parity.py` | 28.0% |
| `src/trend_analysis/data.py` | 29.0% |
| `src/trend_analysis/timefreq.py` | 30.0% |
| `src/trend_analysis/metrics/summary.py` | 33.0% |
| `src/trend_analysis/perf/rolling_cache.py` | 33.0% |
| `src/trend_model/spec.py` | 33.0% |
| `src/trend_analysis/core/metric_cache.py` | 36.0% |
| `src/trend_model/_sitecustomize.py` | 36.0% |
| `src/trend_analysis/_autofix_violation_case2.py` | 39.0% |
| `src/trend_portfolio_app/monte_carlo/engine.py` | 39.0% |
| `src/trend/__init__.py` | 40.0% |
| `src/trend_analysis/metrics/turnover.py` | 41.0% |
| `src/trend_analysis/perf/cache.py` | 41.0% |
| `src/trend_analysis/reporting/__init__.py` | 43.0% |
| `src/trend_analysis/config/model.py` | 44.0% |
| `src/trend_analysis/config/models.py` | 46.0% |
| `src/trend_analysis/util/frequency.py` | 46.0% |
| `src/trend_analysis/config/legacy.py` | 47.0% |
| `src/trend_analysis/signal_presets.py` | 52.0% |
| `src/trend_analysis/_autofix_violation_case3.py` | 53.0% |
| `src/trend_analysis/_autofix_trigger_sample.py` | 54.0% |
| `src/trend_analysis/io/market_data.py` | 54.0% |
| `src/trend_analysis/util/joblib_shim.py` | 54.0% |
| `src/trend_analysis/pipeline.py` | 55.0% |
| `src/trend_analysis/selector.py` | 58.0% |
| `src/trend_model/app.py` | 58.0% |
| `src/trend_analysis/_ci_probe_faults.py` | 60.0% |
| `src/trend_analysis/gui/plugins.py` | 63.0% |
| `src/trend_analysis/signals.py` | 65.0% |
| `src/trend_analysis/weighting.py` | 65.0% |
| `src/trend_analysis/util/missing.py` | 68.0% |
| `src/trend_analysis/__init__.py` | 69.0% |
| `src/trend_analysis/metrics/__init__.py` | 69.0% |
| `src/trend_analysis/risk.py` | 71.0% |
| `src/trend_analysis/automation_multifailure.py` | 75.0% |
| `src/trend_portfolio_app/app.py` | 78.0% |
| `src/trend_analysis/plugins/__init__.py` | 80.0% |
| `src/trend_portfolio_app/data_schema.py` | 83.0% |
| `src/trend_analysis/gui/utils.py` | 91.0% |
| `src/trend_portfolio_app/sim_runner.py` | 94.0% |

_The table reflects the baseline `.coverage` snapshot produced by the sweep above; modules cleared in later targeted runs are still shown here to preserve the historical ordering for triage._【b1981e†L1-L64】

## Targeted improvements
- Re-ran `tests/test_trend_analysis_init_module.py` under coverage to verify that the dataclass guards, lazy imports, and spec proxy logic now exercise every branch; `trend_analysis/__init__.py` reports 96 % coverage (0 statements missed, 4 partial branches).【0c78fe†L1-L5】
- Added `tests/test_run_analysis_entrypoint_modern.py` to simulate CLI invocations, missing/legacy configuration keys, and loader failures without touching the filesystem. The suite combines with the existing keepalive tests to lift `trend_analysis/run_analysis.py` to 99 % coverage (1 defensive statement remains unexecuted).【edae3e†L1-L9】【9bccc5†L1-L5】

## Next steps
- Backfill coverage for the remaining pipeline workhorses surfaced in the soft sweep (e.g., `data.py`, `pipeline.py`, `io/validators.py`, `engine/optimizer.py`) so each clears the 95% bar.
- Break down the long tail of orchestration modules (`signals.py`, `regimes.py`, `cli.py`, etc.) into targeted regression scenarios to avoid the pandas/numpy incompatibilities that caused the legacy suites to fail during the baseline run.
