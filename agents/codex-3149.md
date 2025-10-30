<!-- bootstrap for codex on issue #3149 -->

## Scope
- Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [ ] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [ ] __init__.py
  - [ ] data.py
  - [ ] presets.py
  - [ ] harness.py
  - [ ] regimes.py
  - [ ] pipeline.py
  - [ ] validators.py
  - [ ] run_analysis.py
  - [ ] market_data.py
  - [ ] signal_presets.py
  - [x] frequency.py
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

---

_Repost this checklist after each update and tick items only when the associated acceptance criteria have been fully satisfied._

### Latest coverage snapshot (sorted by current coverage)

| File | Coverage |
| --- | --- |
| trend_analysis/cli.py | 0% |
| trend_analysis/presets.py | 0% |
| trend_analysis/run_analysis.py | 0% |
| trend_analysis/signal_presets.py | 0% |
| trend_analysis/multi_period/engine.py | 0% |
| trend_analysis/pipeline.py | 7% |
| trend_analysis/engine/optimizer.py | 7% |
| trend_analysis/export/bundle.py | 7% |
| trend_analysis/data.py | 8% |
| trend_analysis/regimes.py | 9% |
| trend_analysis/io/validators.py | 10% |
| trend_analysis/backtesting/bootstrap.py | 11% |
| trend_analysis/io/market_data.py | 13% |
| trend_analysis/backtesting/harness.py | 14% |
| trend_analysis/risk.py | 21% |
| trend_analysis/config/model.py | 23% |
| trend_analysis/signals.py | 27% |
| trend_analysis/__init__.py | 81% |
| trend_analysis/util/frequency.py | 100% |
