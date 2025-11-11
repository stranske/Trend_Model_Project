<!-- bootstrap for codex on issue #3470 -->

## Scope
- [ ] Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
  - Latest soft-coverage snapshot (partial):
    - `trend_analysis/io/market_data.py` – 13 %【c88e4e†L75-L79】
    - `trend_analysis/pipeline.py` – 7 %【c88e4e†L110-L113】
    - `trend_analysis/data.py` – 8 %【c88e4e†L100-L103】
    - `trend_analysis/__init__.py` – 68 % before targeted tests; raised to 97 % below.
    - `trend_analysis/presets.py` – 0 % before targeted tests; raised to 96 % below.
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py – 97 % after new soft-coverage tests.【d828b6†L4-L5】
  - [x] presets.py – 96 % after new soft-coverage tests.【d828b6†L5-L6】
  - [ ] data.py
  - [ ] harness.py
  - [ ] regimes.py
  - [ ] pipeline.py
  - [ ] validators.py
  - [ ] run_analysis.py
  - [ ] market_data.py
  - [ ] signal_presets.py
  - [ ] frequency.py
  - [ ] signals.py
  - [ ] bootstrap.p
  - [ ] risk.py
  - [ ] bundle.py
  - [ ] cli.py
  - [ ] optimizer.py
  - [ ] model.py
  - [ ] engine.py

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file.
- [ ] Essential functions for the program have full test coverage.
