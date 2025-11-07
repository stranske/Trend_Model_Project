<!-- bootstrap for codex on issue #3352 -->
# Issue #3352 Coverage Tracker

## Latest targeted coverage run
- `python -m coverage run -m pytest` with focused suites for the files below (see commands in the development log).
- `python -m coverage report -m` for explicit module percentages.

## Module coverage status
- [x] `trend_analysis/__init__.py` – 98% 【61630f†L2-L3】
- [x] `trend_analysis/data.py` – 97% 【61630f†L7-L8】
- [x] `trend_analysis/presets.py` – 96% 【61630f†L13-L14】
- [x] `trend_analysis/backtesting/harness.py` – 100% 【61630f†L4-L5】
- [x] `trend_analysis/regimes.py` – 98% 【61630f†L15-L16】
- [ ] `trend_analysis/pipeline.py` – 87% (needs additional scenarios) 【569f86†L1-L5】
- [x] `trend_analysis/io/validators.py` – 100% 【61630f†L11-L12】
- [x] `trend_analysis/run_analysis.py` – 100% 【61630f†L17-L18】
- [x] `trend_analysis/io/market_data.py` – 97% 【61630f†L9-L10】
- [x] `trend_analysis/market_data.py` – tracked via `io/market_data.py` (see above).
- [x] `trend_analysis/signal_presets.py` – 100% 【61630f†L19-L20】
- [x] `trend_analysis/util/frequency.py` – 100% 【61630f†L21-L22】
- [x] `trend_analysis/signals.py` – 100% 【61630f†L19-L20】
- [x] `trend_analysis/backtesting/bootstrap.py` – 100% 【76a813†L2-L3】
- [x] `trend_analysis/risk.py` – 100% 【61630f†L17-L18】
- [x] `trend_analysis/export/bundle.py` – 99% 【61630f†L9-L10】
- [x] `trend_analysis/cli.py` – 99% 【76a813†L2-L4】
- [x] `trend_analysis/engine/optimizer.py` – 95% 【61630f†L8-L9】
- [x] `trend_analysis/config/model.py` – 98% 【76a813†L4-L5】
- [ ] `trend_analysis/multi_period/engine.py` – 92% (expand helper coverage) 【569f86†L1-L3】

## Outstanding work
- [ ] Raise `trend_analysis/pipeline.py` above 95% (current: 87%).
- [ ] Raise `trend_analysis/multi_period/engine.py` above 95% (current: 92%).
- [ ] Re-run coverage once additional scenarios are in place.

## Acceptance criteria tracking
- [ ] Test coverage exceeds 95% for each file.
- [ ] Essential functions for the program have full test coverage (pending higher coverage for pipeline + multi-period engine).
