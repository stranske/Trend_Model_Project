<!-- bootstrap for codex on issue #3470 -->

## Scope
- [x] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Soft coverage snapshot (targeting `src/trend_analysis` and `src/trend_portfolio_app`) highlighted the lowest-covered modules before remediation. Key gaps included the CLI, pipeline, data loading, validators, and Monte Carlo engine entrypoints.【87f948†L1-L40】
- [x] Increase test coverage incrementally for one set of related issues or one file below at a time.
  - [x] __init__.py – `trend_analysis/__init__.py` now reports 99 % coverage after exercising the dataclass guard, spec proxy, and lazy import paths.【d080ef†L1-L6】
  - [x] data.py – Data ingestion helpers are verified end-to-end with 99 % statement coverage.【c7a653†L1-L6】
  - [x] presets.py – Signal preset configuration logic is fully covered (100 %).【234c4b†L1-L6】
  - [x] harness.py – Backtesting harness utilities sit at 100 % coverage.【6a9e72†L1-L6】
  - [x] regimes.py – Regime detection paths now exercise 99 % of statements and branches.【11de22†L1-L6】
  - [x] pipeline.py – Core pipeline orchestration reaches 99 % coverage across branches.【1e5409†L1-L6】
  - [x] validators.py – IO validators execute all validation paths (100 %).【0953e3†L1-L6】
  - [x] run_analysis.py – CLI dispatch helper runs are fully covered (100 %).【d84978†L1-L6】
  - [x] market_data.py – Market data loaders and validators sit at 100 % coverage.【d338af†L1-L6】
  - [x] signal_presets.py – Signal preset registration achieves 100 % coverage.【72d656†L1-L6】
  - [x] frequency.py – Frequency utilities deliver 100 % coverage.【c9e0bf†L1-L6】
  - [x] signals.py – Signal calculations are fully exercised (100 %).【96173d†L1-L6】
  - [x] bootstrap.py – Block bootstrap engine functions hit 100 % coverage.【a7d426†L1-L6】
  - [x] risk.py – Risk analytics maintain 100 % coverage.【90e11e†L1-L6】
  - [x] bundle.py – Export bundle writer achieves 99 % coverage with only two partial branch misses.【d00e25†L1-L6】
  - [x] cli.py – Command-line interface paths now sit at 98 % coverage with all command groups exercised.【836860†L1-L5】
  - [x] optimizer.py – Optimizer engine logic reports 100 % coverage.【76b46a†L1-L6】
  - [x] model.py – Configuration model loader records 97 % coverage after running targeted validation tests.【32af76†L1-L6】
  - [x] engine.py – Monte Carlo engine reaches 100 % coverage.【c2a708†L1-L6】

## Acceptance criteria
- [x] Test coverage exceeds 95% for each file listed above, based on focused coverage runs for each module.【d080ef†L1-L6】【c7a653†L1-L6】【234c4b†L1-L6】【6a9e72†L1-L6】【11de22†L1-L6】【1e5409†L1-L6】【0953e3†L1-L6】【d84978†L1-L6】【d338af†L1-L6】【72d656†L1-L6】【c9e0bf†L1-L6】【96173d†L1-L6】【a7d426†L1-L6】【90e11e†L1-L6】【d00e25†L1-L6】【836860†L1-L5】【76b46a†L1-L6】【32af76†L1-L6】【c2a708†L1-L6】
- [x] Essential functions for the program have full test coverage through the targeted suites listed above.【d080ef†L1-L6】【c7a653†L1-L6】【234c4b†L1-L6】【6a9e72†L1-L6】【11de22†L1-L6】【1e5409†L1-L6】【0953e3†L1-L6】【d84978†L1-L6】【d338af†L1-L6】【72d656†L1-L6】【c9e0bf†L1-L6】【96173d†L1-L6】【a7d426†L1-L6】【90e11e†L1-L6】【d00e25†L1-L6】【836860†L1-L5】【76b46a†L1-L6】【32af76†L1-L6】【c2a708†L1-L6】
