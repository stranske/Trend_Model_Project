<!-- bootstrap for codex on issue #3470 -->

## Scope
- [x] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.【74bc1e†L1-L23】

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Commands executed (merged via `coverage combine`):
    1. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src/trend_analysis -m pytest tests/trend_analysis`
    2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src/trend_analysis -m pytest tests/test_trend_analysis_cli.py ... tests/test_run_analysis.py`
    3. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src/trend_analysis,src/trend_portfolio_app/monte_carlo -m pytest tests/test_market_data_validation.py ... tests/test_config_model.py`
    4. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src/trend_analysis -m pytest tests/backtesting/test_bootstrap.py tests/test_risk.py tests/test_risk_additional.py tests/test_trend_analysis_package.py tests/test_trend_analysis_init_keepalive.py`
  - Combined report: all targeted modules ≥ 97 % line coverage.【74bc1e†L1-L23】
- [x] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py – 97 % line coverage.【74bc1e†L1-L23】
  - [x] data.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] presets.py – 97 % line coverage.【74bc1e†L1-L23】
  - [x] harness.py – 99 % line coverage.【74bc1e†L1-L23】
  - [x] regimes.py – 99 % line coverage.【74bc1e†L1-L23】
  - [x] pipeline.py – 99 % line coverage.【74bc1e†L1-L23】
  - [x] validators.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] run_analysis.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] market_data.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] signal_presets.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] frequency.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] signals.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] bootstrap.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] risk.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] bundle.py – 99 % line coverage.【74bc1e†L1-L23】
  - [x] cli.py – 99 % line coverage.【74bc1e†L1-L23】
  - [x] optimizer.py – 100 % line coverage.【74bc1e†L1-L23】
  - [x] model.py – 99 % line coverage.【74bc1e†L1-L23】
  - [x] engine.py – 100 % line coverage.【74bc1e†L1-L23】

## Acceptance criteria
- [x] Test coverage exceeds 95% for each file.【74bc1e†L1-L23】
- [x] Essential functions for the program have full test coverage.【74bc1e†L1-L23】
