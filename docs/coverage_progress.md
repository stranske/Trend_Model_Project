# Coverage Improvement Progress

## Scope
Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py
  - [x] data.py
  - [x] presets.py
  - [x] harness.py
  - [x] regimes.py
  - [x] pipeline.py
  - [x] validators.py
  - [x] run_analysis.py
  - [x] market_data.py
  - [x] signal_presets.py
  - [x] frequency.py
  - [x] signals.py
  - [x] bootstrap.py
  - [x] risk.py
  - [x] bundle.py
  - [x] cli.py
  - [x] optimizer.py
  - [ ] model.py
  - [ ] engine.py

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage

## 2025-02-14 coverage verification

- Re-ran focused coverage suites for the keepalive modules, confirming statement/branch coverage ≥ 95 % for every target except `trend_analysis.config.model` and `trend_analysis.multi_period.engine`.【56180e†L1-L6】【c9a303†L1-L6】【70ea75†L1-L6】【4aa0ec†L1-L6】【78dc8a†L1-L6】【ed9a7c†L1-L6】【af192b†L1-L6】【33d62e†L1-L6】【fd03f5†L1-L6】【93ed80†L1-L6】【c01100†L1-L6】【1c9181†L1-L6】【216a6c†L1-L6】【979abe†L1-L6】【785a81†L1-L6】【f07784†L1-L6】【5d6cef†L1-L6】
- Coverage artefacts were produced by combining per-module `coverage run` executions for the relevant pytest suites (e.g. `tests/test_trend_analysis_data.py`, `tests/test_trend_analysis_presets_additional.py`, `tests/test_pipeline.py`, `tests/test_trend_cli.py`, `tests/test_optimizer_constraints_guardrails.py`).【8611de†L1-L9】【7d7546†L1-L3】【884713†L1-L16】【f70891†L1-L34】【8473e5†L1-L52】【56d529†L1-L18】【241a66†L1-L18】【9a07d4†L1-L15】【cb9bb4†L1-L3】【7cb7d4†L1-L13】【1618c1†L1-L13】【60a237†L1-L3】【295207†L1-L11】【df0e0c†L1-L3】【81c7aa†L1-L9】【65f3db†L1-L3】

| Module | Coverage |
| --- | ---: |
| `src/trend_analysis/__init__.py` | 98 % |
| `src/trend_analysis/data.py` | 98 % |
| `src/trend_analysis/presets.py` | 100 % |
| `src/trend_analysis/backtesting/harness.py` | 100 % |
| `src/trend_analysis/regimes.py` | 100 % |
| `src/trend_analysis/pipeline.py` | 99 % |
| `src/trend_analysis/io/validators.py` | 100 % |
| `src/trend_analysis/run_analysis.py` | 100 % |
| `src/trend_analysis/io/market_data.py` | 99 % |
| `src/trend_analysis/signal_presets.py` | 100 % |
| `src/trend_analysis/util/frequency.py` | 100 % |
| `src/trend_analysis/signals.py` | 100 % |
| `src/trend_analysis/backtesting/bootstrap.py` | 100 % |
| `src/trend_analysis/risk.py` | 100 % |
| `src/trend_analysis/export/bundle.py` | 99 % |
| `src/trend_analysis/cli.py` | 98 % |
| `src/trend_analysis/engine/optimizer.py` | 100 % |

### Remaining sub-95 % targets

- `src/trend_analysis/config/model.py` — 54 % coverage; missing scenarios span path resolution error handling and fallback coercion branches.【29ded7†L1-L5】
- `src/trend_analysis/multi_period/engine.py` — 87 % coverage; gaps include turnover rescale enforcement, reseeding, and schedule boundary paths.【140864†L1-L5】

## Next steps
- Implement regression tests that drive the outstanding branches in `trend_analysis.config.model` and `trend_analysis.multi_period.engine` so both modules exceed the 95 % threshold.
