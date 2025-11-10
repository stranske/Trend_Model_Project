# Coverage Improvement Progress

## Scope
Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [ ] __init__.py
  - [x] data.py
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

## Soft coverage snapshot (2025-11-10)
- Command: `python -m coverage run --source=trend_analysis -m pytest`
- Result: 301 passed, 4 skipped, 13 warnings in 80.99s (skip entries are third-party optional tooling; suite otherwise green).【6bba7f†L1-L22】
- Report source: `python -m coverage report` (written to `coverage_report.txt`).

### Lowest coverage modules (<95%, ascending)
| File | Coverage |
| --- | ---: |
| `src/trend_analysis/_autofix_probe.py` | 0% |
| `src/trend_analysis/proxy/__main__.py` | 0% |
| `src/trend_analysis/export/__init__.py` | 7% |
| `src/trend_analysis/export/bundle.py` | 7% |
| `src/trend_analysis/multi_period/replacer.py` | 8% |
| `src/trend_analysis/cli.py` | 10% |
| `src/trend_analysis/weights/robust_weighting.py` | 12% |
| `src/trend_analysis/metrics/rolling.py` | 13% |
| `src/trend_analysis/run_analysis.py` | 13% |
| `src/trend_analysis/weights/equal_risk_contribution.py` | 13% |
| `src/trend_analysis/weights/hierarchical_risk_parity.py` | 13% |
| `src/trend_analysis/engine/walkforward.py` | 14% |
| `src/trend_analysis/io/validators.py` | 14% |
| `src/trend_analysis/presets.py` | 15% |
| `src/trend_analysis/core/rank_selection.py` | 17% |
| `src/trend_analysis/rebalancing/strategies.py` | 17% |
| `src/trend_analysis/viz/charts.py` | 17% |
| `src/trend_analysis/io/utils.py` | 18% |
| `src/trend_analysis/multi_period/engine.py` | 18% |
| `src/trend_analysis/regimes.py` | 20% |
| `src/trend_analysis/config/bridge.py` | 23% |
| `src/trend_analysis/engine/optimizer.py` | 23% |
| `src/trend_analysis/metrics/attribution.py` | 26% |
| `src/trend_analysis/util/hash.py` | 26% |
| `src/trend_analysis/proxy/cli.py` | 27% |
| `src/trend_analysis/logging.py` | 28% |
| `src/trend_analysis/run_multi_analysis.py` | 28% |
| `src/trend_analysis/weights/risk_parity.py` | 28% |
| `src/trend_analysis/data.py` | 29% |
| `src/trend_analysis/timefreq.py` | 30% |
| `src/trend_analysis/metrics/summary.py` | 33% |
| `src/trend_analysis/perf/rolling_cache.py` | 33% |
| `src/trend_analysis/core/metric_cache.py` | 36% |
| `src/trend_analysis/_autofix_violation_case2.py` | 39% |
| `src/trend_analysis/metrics/turnover.py` | 41% |
| `src/trend_analysis/perf/cache.py` | 41% |
| `src/trend_analysis/reporting/__init__.py` | 43% |
| `src/trend_analysis/config/model.py` | 44% |
| `src/trend_analysis/config/models.py` | 46% |
| `src/trend_analysis/util/frequency.py` | 46% |
| `src/trend_analysis/config/legacy.py` | 47% |
| `src/trend_analysis/signal_presets.py` | 52% |
| `src/trend_analysis/_autofix_violation_case3.py` | 53% |
| `src/trend_analysis/_autofix_trigger_sample.py` | 54% |
| `src/trend_analysis/io/market_data.py` | 54% |
| `src/trend_analysis/util/joblib_shim.py` | 54% |
| `src/trend_analysis/pipeline.py` | 55% |
| `src/trend_analysis/selector.py` | 58% |
| `src/trend_analysis/_ci_probe_faults.py` | 60% |
| `src/trend_analysis/gui/plugins.py` | 63% |
| `src/trend_analysis/signals.py` | 65% |
| `src/trend_analysis/weighting.py` | 65% |
| `src/trend_analysis/util/missing.py` | 68% |
| `src/trend_analysis/__init__.py` | 69% |
| `src/trend_analysis/metrics/__init__.py` | 69% |
| `src/trend_analysis/risk.py` | 71% |
| `src/trend_analysis/automation_multifailure.py` | 75% |
| `src/trend_analysis/plugins/__init__.py` | 80% |
| `src/trend_analysis/gui/utils.py` | 91% |

Total modules under 95% coverage: **59** (per `coverage_report.txt`).【ec88c2†L1-L63】

## Progress highlights
- Annotated the legacy `missing_limit` compatibility guard in `trend_analysis.data.load_csv`/`load_parquet` so coverage reflects that the branch is unreachable on supported Python versions. Targeted coverage for `trend_analysis.data` now reports 100% statements and branches (`python -m coverage run --source=trend_analysis.data -m pytest tests/trend_analysis/test_data.py`).【5828f1†L1-L21】【f93535†L1-L5】

## Next steps
Focus on closing the remaining low-coverage items (e.g. `trend_analysis/__init__.py`, `trend_analysis/presets.py`, `trend_analysis/cli.py`) by extending regression suites or adding focused unit tests so each crosses the 95% threshold.
