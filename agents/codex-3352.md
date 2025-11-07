<!-- bootstrap for codex on issue #3352 -->

# Coverage progress for Trend Analysis modules

## Latest verification runs

| Module | Coverage | Evidence |
| --- | --- | --- |
| `trend_analysis/__init__.py` | 98% | `coverage report src/trend_analysis/__init__.py`【42a41c†L1-L6】 |
| `trend_analysis/data.py` | 98% | `coverage report src/trend_analysis/data.py`【c3a125†L1-L6】 |
| `trend_analysis/presets.py` | 96% | `coverage report src/trend_analysis/presets.py`【3bda21†L1-L6】 |
| `trend_analysis/backtesting/harness.py` | 100% | `coverage report src/trend_analysis/backtesting/harness.py`【d97e1f†L1-L6】 |
| `trend_analysis/regimes.py` | 98% | `coverage report src/trend_analysis/regimes.py`【c83773†L1-L6】 |
| `trend_analysis/io/validators.py` | 100% | `coverage report src/trend_analysis/io/validators.py`【8e7432†L1-L6】 |
| `trend_analysis/run_analysis.py` | 100% | `coverage report src/trend_analysis/run_analysis.py`【8a9b03†L1-L6】 |
| `trend_analysis/io/market_data.py` | 97% | `coverage report src/trend_analysis/io/market_data.py`【66a821†L1-L6】 |
| `trend_analysis/signal_presets.py` | 100% | `coverage report src/trend_analysis/signal_presets.py`【99cf89†L1-L6】 |
| `trend_analysis/util/frequency.py` | 100% | `coverage report src/trend_analysis/util/frequency.py`【dc16e6†L1-L6】 |
| `trend_analysis/signals.py` | 100% | `coverage report src/trend_analysis/signals.py`【266d53†L1-L6】 |
| `trend_analysis/backtesting/bootstrap.py` | 100% | `coverage report src/trend_analysis/backtesting/bootstrap.py`【5fcc49†L1-L6】 |
| `trend_analysis/risk.py` | 100% | `coverage report src/trend_analysis/risk.py`【08a8ff†L1-L6】 |
| `trend_analysis/export/bundle.py` | 99% | `coverage report src/trend_analysis/export/bundle.py`【67b577†L1-L6】 |
| `trend_analysis/cli.py` | 99% | `coverage report src/trend_analysis/cli.py`【3dcd80†L1-L6】 |
| `trend_analysis/engine/optimizer.py` | 95% | `coverage report src/trend_analysis/engine/optimizer.py`【5d3b07†L1-L6】 |
| `trend_analysis/config/model.py` | 98% | `coverage report src/trend_analysis/config/model.py`【95d432†L1-L6】 |

## Outstanding targets

* `trend_analysis/pipeline.py` — current focused test batch reached 83 % statement coverage; needs targeted scenarios for untested branches (e.g. preprocessing summary variants, advanced selection paths, caching fallbacks).
* `trend_analysis/multi_period/engine.py` — combined multi-period engine suites sit at 91 %; additional cases required for gap-handling and warmup/turnover branches outlined in missing-line report.【edff7e†L1-L4】【368a07†L1-L6】

## Next steps

1. Design lightweight fixtures to exercise pipeline’s monthly cadence branch, config unwrap helpers, and weight-engine fallback to lift coverage above 95 % without invoking the full simulation stack.
2. Extend multi-period engine tests to hit window-derivation fallbacks and portfolio diagnostics highlighted by the missing-line ranges.
