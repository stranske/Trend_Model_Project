# Soft coverage findings

Generated from `python -m coverage run --source=trend_analysis -m pytest` targeting the focused suite documented in `docs/coverage_progress.md`.

| Coverage | Missing | Statements | File |
|----------|---------|------------|------|
| 0.00% | 4 | 4 | `src/trend_analysis/_autofix_probe.py` |
| 0.00% | 13 | 13 | `src/trend_analysis/_autofix_trigger_sample.py` |
| 0.00% | 19 | 19 | `src/trend_analysis/_autofix_violation_case2.py` |
| 0.00% | 15 | 15 | `src/trend_analysis/_autofix_violation_case3.py` |
| 0.00% | 15 | 15 | `src/trend_analysis/_ci_probe_faults.py` |
| 0.00% | 10 | 10 | `src/trend_analysis/_typing.py` |
| 0.00% | 24 | 24 | `src/trend_analysis/api_server/__init__.py` |
| 0.00% | 3 | 3 | `src/trend_analysis/api_server/__main__.py` |
| 0.00% | 4 | 4 | `src/trend_analysis/automation_multifailure.py` |
| 0.00% | 277 | 277 | `src/trend_analysis/cli.py` |
| 0.00% | 24 | 24 | `src/trend_analysis/config/bridge.py` |
| 0.00% | 47 | 47 | `src/trend_analysis/config/legacy.py` |
| 0.00% | 49 | 49 | `src/trend_analysis/core/metric_cache.py` |
| 0.00% | 5 | 5 | `src/trend_analysis/gui/__init__.py` |
| 0.00% | 340 | 340 | `src/trend_analysis/gui/app.py` |
| 0.00% | 15 | 15 | `src/trend_analysis/gui/plugins.py` |
| 0.00% | 17 | 17 | `src/trend_analysis/gui/store.py` |
| 0.00% | 27 | 27 | `src/trend_analysis/gui/utils.py` |
| 0.00% | 2 | 2 | `src/trend_analysis/multi_period/__init__.py` |
| 0.00% | 536 | 536 | `src/trend_analysis/multi_period/engine.py` |
| 0.00% | 83 | 83 | `src/trend_analysis/multi_period/replacer.py` |
| 0.00% | 25 | 25 | `src/trend_analysis/multi_period/scheduler.py` |
| 0.00% | 40 | 40 | `src/trend_analysis/plugins/__init__.py` |
| 0.00% | 2 | 2 | `src/trend_analysis/proxy/__init__.py` |
| 0.00% | 3 | 3 | `src/trend_analysis/proxy/__main__.py` |
| 0.00% | 24 | 24 | `src/trend_analysis/proxy/cli.py` |
| 0.00% | 95 | 95 | `src/trend_analysis/proxy/server.py` |
| 0.00% | 8 | 8 | `src/trend_analysis/rebalancing.py` |
| 0.00% | 2 | 2 | `src/trend_analysis/rebalancing/__init__.py` |
| 0.00% | 160 | 160 | `src/trend_analysis/rebalancing/strategies.py` |
| 0.00% | 14 | 14 | `src/trend_analysis/reporting/__init__.py` |
| 0.00% | 32 | 32 | `src/trend_analysis/run_multi_analysis.py` |
| 0.00% | 39 | 39 | `src/trend_analysis/selector.py` |
| 0.00% | 42 | 42 | `src/trend_analysis/signal_presets.py` |
| 0.00% | 14 | 14 | `src/trend_analysis/typing.py` |
| 0.00% | 2 | 2 | `src/trend_analysis/viz/__init__.py` |
| 0.00% | 66 | 66 | `src/trend_analysis/viz/charts.py` |
| 0.00% | 112 | 112 | `src/trend_analysis/weighting.py` |
| 0.00% | 5 | 5 | `src/trend_analysis/weights/__init__.py` |
| 0.00% | 66 | 66 | `src/trend_analysis/weights/equal_risk_contribution.py` |
| 0.00% | 83 | 83 | `src/trend_analysis/weights/hierarchical_risk_parity.py` |
| 0.00% | 28 | 28 | `src/trend_analysis/weights/risk_parity.py` |
| 0.00% | 160 | 160 | `src/trend_analysis/weights/robust_weighting.py` |
| 11.54% | 144 | 170 | `src/trend_analysis/engine/optimizer.py` |
| 12.82% | 26 | 31 | `src/trend_analysis/metrics/rolling.py` |
| 13.64% | 169 | 208 | `src/trend_analysis/backtesting/harness.py` |
| 13.90% | 130 | 161 | `src/trend_analysis/engine/walkforward.py` |
| 16.32% | 386 | 503 | `src/trend_analysis/core/rank_selection.py` |
| 19.01% | 206 | 277 | `src/trend_analysis/regimes.py` |
| 25.53% | 24 | 35 | `src/trend_analysis/metrics/attribution.py` |
| 27.83% | 54 | 85 | `src/trend_analysis/logging.py` |
| 29.55% | 21 | 34 | `src/trend_analysis/timefreq.py` |
| 32.61% | 45 | 74 | `src/trend_analysis/perf/rolling_cache.py` |
| 33.33% | 14 | 21 | `src/trend_analysis/metrics/summary.py` |
| 38.66% | 131 | 248 | `src/trend_analysis/config/model.py` |
| 38.84% | 346 | 611 | `src/trend_analysis/pipeline.py` |
| 41.18% | 8 | 15 | `src/trend_analysis/metrics/turnover.py` |
| 43.70% | 449 | 866 | `src/trend_analysis/export/__init__.py` |
| 44.33% | 194 | 404 | `src/trend_analysis/config/models.py` |
| 53.85% | 6 | 13 | `src/trend_analysis/util/joblib_shim.py` |
| 55.91% | 39 | 101 | `src/trend_analysis/risk.py` |
| 56.13% | 74 | 193 | `src/trend_analysis/metrics/__init__.py` |
| 65.33% | 17 | 57 | `src/trend_analysis/signals.py` |
| 66.17% | 126 | 407 | `src/trend_analysis/io/market_data.py` |
| 78.79% | 13 | 85 | `src/trend_analysis/api.py` |
| 79.31% | 9 | 63 | `src/trend_analysis/backtesting/bootstrap.py` |
| 85.71% | 3 | 32 | `src/trend_analysis/util/hash.py` |
| 91.67% | 7 | 116 | `src/trend_analysis/util/missing.py` |

Total files below 95%: 68
