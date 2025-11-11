<!-- bootstrap for codex on issue #3470 -->

## Scope
- [ ] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest`【abb9c9†L1-L1】
  - Current high-priority modules below 95 % line coverage (sorted from lowest coverage upward):
    1. `trend_analysis/export/bundle.py` – 7 %【37fc58†L61-L63】
    2. `trend_analysis/cli.py` – 10 %【37fc58†L38-L40】
    3. `trend_analysis/regimes.py` – 11 %【37fc58†L93-L95】
    4. `trend_analysis/run_analysis.py` – 13 %【37fc58†L97-L98】
    5. `trend_analysis/io/validators.py` – 14 %【37fc58†L53-L55】
    6. `trend_analysis/presets.py` – 15 %【37fc58†L87-L89】
    7. `trend_analysis/engine/optimizer.py` – 22 %【37fc58†L71-L73】
    8. `trend_analysis/pipeline.py` – 38 %【37fc58†L79-L83】
    9. `trend_portfolio_app/monte_carlo/engine.py` – 39 %【37fc58†L121-L122】
    10. `trend_analysis/config/model.py` – 44 %【37fc58†L75-L77】
    11. `trend_analysis/util/frequency.py` – 46 %【37fc58†L109-L110】
    12. `trend_analysis/signal_presets.py` – 52 %【37fc58†L101-L103】
    13. `trend_analysis/io/market_data.py` – 54 %【37fc58†L57-L60】
    14. `trend_analysis/signals.py` – 65 %【37fc58†L105-L106】
    15. `trend_analysis/risk.py` – 70 %【37fc58†L95-L96】
    16. `trend_analysis/backtesting/bootstrap.py` – 100 % (already above target)【37fc58†L33-L34】
    17. `trend_analysis/backtesting/harness.py` – 100 % (already above target)【37fc58†L35-L36】
  - Modules lifted above the threshold during this iteration:
    - `trend_analysis/__init__.py` – 98 % (met)【ddf65b†L1-L5】
    - `trend_analysis/data.py` – 99 % (met)【1198e5†L1-L5】
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py – 98 % line coverage after targeted tests【ddf65b†L1-L5】
    - Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=trend_analysis -m pytest tests/test_trend_analysis_init.py`【fa9f59†L1-L1】
  - [x] data.py – 99 % line coverage (parse-hint logging verified).【aa2552†L1-L1】【1198e5†L1-L5】
    - Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=trend_analysis.data -m pytest tests/test_trend_analysis_data.py tests/test_trend_analysis_data_additional.py`【aa2552†L1-L1】
    - Report: `python -m coverage report -m src/trend_analysis/data.py`【1198e5†L1-L5】
  - [ ] presets.py – 15 % line coverage【37fc58†L87-L89】
  - [x] harness.py – 100 % line coverage (already satisfies target)【37fc58†L35-L36】
  - [ ] regimes.py – 11 % line coverage【37fc58†L93-L95】
  - [ ] pipeline.py – 38 % line coverage【37fc58†L79-L83】
  - [ ] validators.py – 14 % line coverage【37fc58†L53-L55】
  - [ ] run_analysis.py – 13 % line coverage【37fc58†L97-L98】
  - [ ] market_data.py – 54 % line coverage【37fc58†L57-L60】
  - [ ] signal_presets.py – 52 % line coverage【37fc58†L101-L103】
  - [ ] frequency.py – 46 % line coverage【37fc58†L109-L110】
  - [ ] signals.py – 65 % line coverage【37fc58†L105-L106】
  - [x] bootstrap.py – 100 % line coverage (already satisfies target)【37fc58†L33-L34】
  - [ ] risk.py – 70 % line coverage【37fc58†L95-L96】
  - [ ] bundle.py – 7 % line coverage【37fc58†L61-L63】
  - [ ] cli.py – 10 % line coverage【37fc58†L38-L40】
  - [ ] optimizer.py – 22 % line coverage【37fc58†L71-L73】
  - [ ] model.py – 44 % line coverage【37fc58†L75-L77】
  - [ ] engine.py – 39 % line coverage【37fc58†L121-L122】

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage
