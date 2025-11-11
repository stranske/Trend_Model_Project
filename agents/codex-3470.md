<!-- bootstrap for codex on issue #3470 -->

## Scope
- [x] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Targeted coverage runs confirmed each module now exceeds 95%:
    - `trend_analysis/__init__.py` – 96 %【70c5e2†L1-L22】
    - `trend_analysis/data.py` – 98 %【70c5e2†L1-L22】
    - `trend_analysis/presets.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/backtesting/harness.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/regimes.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/pipeline.py` – 99 %【70c5e2†L1-L22】
    - `trend_analysis/io/validators.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/run_analysis.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/io/market_data.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/signal_presets.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/util/frequency.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/signals.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/backtesting/bootstrap.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/risk.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/export/bundle.py` – 99 %【70c5e2†L1-L22】
    - `trend_analysis/cli.py` – 99 %【70c5e2†L1-L22】
    - `trend_analysis/engine/optimizer.py` – 100 %【70c5e2†L1-L22】
    - `trend_analysis/config/model.py` – 99 %【70c5e2†L1-L22】
    - `trend_portfolio_app/monte_carlo/engine.py` – 100 %【70c5e2†L1-L22】
- [x] Increase test coverage incrementally for one set of related issues or 1 file below at a time
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
  - [x] bootstrap.p
  - [x] risk.py
  - [x] bundle.py
  - [x] cli.py
  - [x] optimizer.py
  - [x] model.py
  - [x] engine.py

## Acceptance criteria
- [x] Test coverage exceeds 95% for each file
- [x] Essential functions for the program have full test coverage

Latest coverage command:

```
python -m coverage report -m src/trend_analysis/__init__.py src/trend_analysis/data.py src/trend_analysis/presets.py src/trend_analysis/backtesting/harness.py src/trend_analysis/regimes.py src/trend_analysis/pipeline.py src/trend_analysis/io/validators.py src/trend_analysis/run_analysis.py src/trend_analysis/io/market_data.py src/trend_analysis/signal_presets.py src/trend_analysis/util/frequency.py src/trend_analysis/signals.py src/trend_analysis/backtesting/bootstrap.py src/trend_analysis/risk.py src/trend_analysis/export/bundle.py src/trend_analysis/cli.py src/trend_analysis/engine/optimizer.py src/trend_analysis/config/model.py src/trend_portfolio_app/monte_carlo/engine.py
```

All outputs for the command above are recorded in `coverage-summary.md`.【F:coverage-summary.md†L1-L27】
