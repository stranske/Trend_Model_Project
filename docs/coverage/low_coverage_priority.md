# Coverage priorities for trend_analysis modules

The table below captures the current "soft" coverage snapshot gathered from running `pytest tests/test_trend_analysis_package_init.py --cov=trend_analysis --cov-report=term-missing`.

| File | Coverage | Notes |
| --- | --- | --- |
| `src/trend_analysis/cli.py` | 0% | Command-line entry point remains untested. |
| `src/trend_analysis/presets.py` | 0% | Preset registry logic entirely uncovered. |
| `src/trend_analysis/run_analysis.py` | 0% | Pipeline wrapper lacks coverage. |
| `src/trend_analysis/signal_presets.py` | 0% | Signal preset definitions not exercised. |
| `src/trend_analysis/multi_period/engine.py` | 0% | Multi-period engine core lacks tests. |
| `src/trend_analysis/export/bundle.py` | 7% | Bundle exporter has minimal coverage. |
| `src/trend_analysis/pipeline.py` | 7% | Main pipeline orchestration largely untested in this run. |
| `src/trend_analysis/engine/optimizer.py` | 7% | Optimizer logic requires coverage. |
| `src/trend_analysis/data.py` | 8% | Data loaders and helpers mostly uncovered. |
| `src/trend_analysis/regimes.py` | 9% | Regime detection logic needs tests. |
| `src/trend_analysis/io/validators.py` | 10% | Input validators have sparse coverage. |
| `src/trend_analysis/backtesting/bootstrap.py` | 11% | Bootstrap routines minimally covered. |
| `src/trend_analysis/io/market_data.py` | 13% | Market data ingestion paths under-covered. |
| `src/trend_analysis/backtesting/harness.py` | 14% | Harness utilities need more coverage. |
| `src/trend_analysis/util/frequency.py` | 20% | Frequency utilities partially covered. |
| `src/trend_analysis/risk.py` | 21% | Risk helpers missing tests. |
| `src/trend_analysis/config/model.py` | 23% | Config model definitions under-covered. |
| `src/trend_analysis/signals.py` | 27% | Signal generation functions require tests. |
| `src/trend_analysis/__init__.py` | 100% | Achieved full coverage via targeted tests. |

