# Soft Coverage Sweep — Round 1

## Command
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=trend_analysis -m pytest -q
python -m coverage report -m > coverage_report.txt
```

The raw `coverage report -m` output is available locally via `coverage_report.txt` (ignored in git to avoid churn). Re-run the command above to regenerate it.

## Lowest-Coverage Targets (<95%)

| Rank | Module | Coverage | Notes |
| --- | --- | ---: | --- |
| 1 | `src/trend_analysis/export/bundle.py` | 7% | Export bundle orchestrator virtually untested. |
| 2 | `src/trend_analysis/cli.py` | 10% | CLI entrypoint / command plumbing mostly uncovered. |
| 3 | `src/trend_analysis/run_analysis.py` | 13% | CLI-run workflow shim lacks regression coverage. |
| 4 | `src/trend_analysis/io/validators.py` | 14% | Input validation guards barely exercised. |
| 5 | `src/trend_analysis/presets.py` | 15% | Preset resolution logic missing tests. |
| 6 | `src/trend_analysis/multi_period/engine.py` | 18% | Multi-period engine integration mostly untested. |
| 7 | `src/trend_analysis/regimes.py` | 20% | Regime detection and caching untested. |
| 8 | `src/trend_analysis/engine/optimizer.py` | 23% | Optimizer safeguards untested. |
| 9 | `src/trend_analysis/data.py` | 29% | Data ingestion/normalization lacking coverage. |
| 10 | `src/trend_analysis/config/model.py` | 44% | Model config schema has large uncovered surface. |
| 11 | `src/trend_analysis/util/frequency.py` | 46% | Frequency helper still needs tests. |
| 12 | `src/trend_analysis/signal_presets.py` | 52% | Preset wrappers need coverage. |
| 13 | `src/trend_analysis/io/market_data.py` | 54% | Market data validator missing regression. |
| 14 | `src/trend_analysis/pipeline.py` | 55% | Pipeline orchestration partially exercised. |
| 15 | `src/trend_analysis/signals.py` | 65% | Signal builders require additional coverage. |
| 16 | `src/trend_analysis/__init__.py` | 69% | Package init re-export logic still uncovered. |
| 17 | `src/trend_analysis/risk.py` | 71% | Risk calculators need tests. |
| 18 | `src/trend_analysis/backtesting/bootstrap.py` | 100% | Fully covered; no further action required. |
| 19 | `src/trend_analysis/backtesting/harness.py` | 100% | Fully covered; no further action required. |

These priorities align with the keepalive task list; subsequent rounds will target one module at a time, starting with `export/bundle.py` and `cli.py`.

## Progress

- Added `tests/test_run_analysis_cli_soft_round1.py` to exercise both detailed and summary code paths in `trend_analysis.run_analysis.main`, covering legacy `nan_*` configuration keys, CSV loader argument translation, successful export delegation, and the error guardrails for missing inputs.
- Targeted coverage run confirms the module now reports **96%** line coverage (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=trend_analysis.run_analysis -m pytest tests/test_run_analysis_cli_soft_round1.py -q` followed by `python -m coverage report -m src/trend_analysis/run_analysis.py`).
