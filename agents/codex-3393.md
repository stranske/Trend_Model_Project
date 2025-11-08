<!-- bootstrap for codex on issue #3393 -->

## Scope
- Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered. 【a7b6cd†L1-L35】【ad8a52†L1-L87】
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] `__init__.py` — 97% statement coverage in the latest soft sweep.【ad8a52†L3-L3】
  - [x] `data.py` — 97% with four statements and seven branch parts still outstanding.【ad8a52†L27-L27】
  - [x] `presets.py` — 99% with a single remaining statement gap.【ad8a52†L55-L55】
  - [ ] `harness.py` — 14% across `backtesting/harness.py`; major logic remains untested.【ad8a52†L15-L16】
  - [ ] `regimes.py` — 19% coverage with broad untested branches.【ad8a52†L63-L63】
  - [ ] `pipeline.py` — 39% and 200 uncovered branches, so requires extensive suites.【ad8a52†L53-L53】
  - [x] `validators.py` — 100% statement/branch coverage confirmed.【ad8a52†L41-L41】
  - [x] `run_analysis.py` — 100% coverage from existing CLI regression harnesses.【ad8a52†L66-L66】
  - [ ] `market_data.py` — 66% with heavy gaps in validation error paths.【ad8a52†L39-L39】
  - [ ] `signal_presets.py` — 0% currently; needs initial test scaffolding.【ad8a52†L69-L69】
  - [x] `frequency.py` — 100% from util frequency suites.【ad8a52†L74-L74】
  - [ ] `signals.py` — 65% with multiple indicator combinations untested.【ad8a52†L70-L70】
  - [ ] `bootstrap.py` — 79% across `backtesting/bootstrap.py`; scenario coverage still missing.【ad8a52†L15-L15】
  - [ ] `risk.py` — 56% with branch-heavy sections uncovered.【ad8a52†L65-L65】
  - [x] `bundle.py` — 99% after existing export bundle suites.【ad8a52†L32-L32】
  - [x] `cli.py` — 98% coverage confirmed via targeted CLI suite.
  - [ ] `optimizer.py` — 12% for `engine/optimizer.py`; optimisation paths lack coverage.【ad8a52†L29-L29】
  - [ ] `model.py` — 39% for `config/model.py`, still far from target.【ad8a52†L21-L21】
  - [ ] `engine.py` — 14% for `engine/walkforward.py`; walk-forward coordination remains untested.【ad8a52†L30-L30】

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file.
- [ ] Essential functions for the program have full test coverage.

## Soft coverage sweep (2025-10-13)
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run --source=trend_analysis -m pytest \
  tests/test_validators.py \
  tests/test_io_validators_additional.py \
  tests/test_io_validators_extra.py \
  tests/test_io_validators_negative_paths.py \
  tests/test_io_utils.py \
  test_upload_app.py \
  tests/test_export_bundle.py \
  tests/test_run_analysis_cli_branches.py \
  tests/test_run_analysis_cli_export.py \
  tests/test_default_export.py \
  tests/test_trend_analysis_presets.py \
  tests/test_trend_analysis_presets_additional.py \
  tests/test_trend_analysis_data.py \
  tests/test_trend_analysis_data_additional.py \
  tests/test_trend_analysis_init.py \
  tests/test_trend_analysis_init_extra.py \
  tests/test_trend_analysis_cli_main.py \
  tests/unit/util/test_frequency_comprehensive.py \
  tests/test_frequency_missing.py \
  tests/test_util_frequency_additional.py \
  tests/test_util_frequency_missing.py
coverage report -m > coverage-soft.txt
```
- 250 tests passed (24 warnings) in 12.68s, producing a consolidated coverage report for prioritisation.【a7b6cd†L1-L35】【7797c1†L1-L74】
- Targeted CLI suite (`tests/test_trend_analysis_cli_main.py`) now executes 21 focused scenarios and reports 98% coverage for `trend_analysis/cli.py`, confirming the module meets the ≥95% goal.【2a8baf†L1-L12】

### Lowest-coverage modules (<95%)
| Coverage | Module | Notes |
| --- | --- | --- |
| 0% | `trend_analysis/signal_presets.py` | Signal preset defaults lack regression coverage; need to port scenarios from manual QA.【ad8a52†L69-L69】 |
| 12% | `trend_analysis/engine/optimizer.py` | Optimiser construction and error handling remain untested; design focused unit specs before wiring into pipelines.【ad8a52†L29-L29】 |
| 14% | `trend_analysis/backtesting/harness.py` | Bootstrapping and walk-forward loops mostly uncovered; build fixtures around synthetic windows.【ad8a52†L15-L16】 |
| 14% | `trend_analysis/engine/walkforward.py` | Coordination logic and sliding window state transitions require dedicated suites.【ad8a52†L30-L30】 |
| 19% | `trend_analysis/regimes.py` | Regime detection heuristics and configuration parsing still missing tests.【ad8a52†L63-L63】 |
| 35% | `trend_analysis/export/__init__.py` | Bulk exporter options cover numerous branches; identify critical paths to exercise iteratively.【ad8a52†L31-L31】 |
| 39% | `trend_analysis/pipeline.py` | Core pipeline remains the largest gap with 346 uncovered statements and 200 uncovered branches.【ad8a52†L53-L53】 |
| 39% | `trend_analysis/config/model.py` | Model schema coercion and defaults remain lightly tested; consider parameterised fixtures.【ad8a52†L21-L21】 |
| 56% | `trend_analysis/risk.py` | Risk metric aggregations have numerous untested branches, especially around fallback handling.【ad8a52†L65-L65】 |
| 65% | `trend_analysis/signals.py` | Multiple indicator combinations and error paths missing from the suite.【ad8a52†L70-L70】 |
| 66% | `trend_analysis/io/market_data.py` | Validation and coercion fallbacks still need full coverage, particularly around missing data policies.【ad8a52†L39-L39】 |

### Next steps
1. Build focused regression suites for `backtesting/harness.py` and `engine/walkforward.py` that simulate small synthetic universes to exercise bootstrap iterations, state resets, and error paths.
2. Extend market data validation tests to cover missing-policy overrides and tolerance edge cases, raising coverage for `market_data.py` while hardening essential IO functionality.
3. Continue planning regression suites for CLI-adjacent configuration files (e.g., `config/model.py`) to leverage the new targeted CLI run as scaffolding once higher-priority modules are addressed.
