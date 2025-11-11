<!-- bootstrap for codex on issue #3470 -->

## Scope
- [x] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Targeted coverage runs confirmed each module now exceeds 95%:
    - `trend_analysis/__init__.py` – 99 %【ec3a7b†L1-L6】
    - `trend_analysis/data.py` – 99 %【8a13f9†L1-L4】
    - `trend_analysis/presets.py` – 100 %【f5fdf3†L1-L4】
    - `trend_analysis/backtesting/harness.py` – 100 %【83ab41†L1-L4】
    - `trend_analysis/regimes.py` – 99 %【163de3†L1-L4】
    - `trend_analysis/pipeline.py` – 99 %【6dbcd2†L1-L5】
    - `trend_analysis/io/validators.py` – 100 %【59d676†L1-L4】
    - `trend_analysis/run_analysis.py` – 99 %【aecd7d†L1-L5】
    - `trend_analysis/io/market_data.py` – 100 %【1bff4f†L1-L4】
    - `trend_analysis/signal_presets.py` – 100 %【7ce21c†L1-L4】
    - `trend_analysis/util/frequency.py` – 100 %【562b83†L1-L4】
    - `trend_analysis/signals.py` – 100 %【3891ef†L1-L4】
    - `trend_analysis/backtesting/bootstrap.py` – 100 %【948a6e†L1-L4】
    - `trend_analysis/risk.py` – 100 %【1d9711†L1-L4】
    - `trend_analysis/export/bundle.py` – 99 %【597dac†L1-L4】
    - `trend_analysis/cli.py` – 99 %【7be9e6†L1-L4】
    - `trend_analysis/engine/optimizer.py` – 100 %【1b1469†L1-L4】
    - `trend_analysis/config/model.py` – 99 %【a13f70†L1-L4】
    - `trend_portfolio_app/monte_carlo/engine.py` – 100 %【92ae40†L1-L4】
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
