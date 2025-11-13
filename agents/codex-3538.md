<!-- bootstrap for codex on issue #3538 -->

## Scope
- Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Tasks
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered. *(Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest` — run interrupted manually after producing a usable report.)*
- [ ] Increase test coverage incrementally for one set of related issues or 1 file below at a time
  - [x] __init__.py — src/trend_analysis/__init__.py (97%)
  - [x] data.py — src/trend_analysis/data.py (98%)
  - [ ] presets.py — src/trend_analysis/presets.py (15%)
  - [x] harness.py — src/trend_analysis/backtesting/harness.py (100%)
  - [x] regimes.py — src/trend_analysis/regimes.py (99%)
  - [ ] pipeline.py — src/trend_analysis/pipeline.py (55%)
  - [ ] validators.py — src/trend_analysis/io/validators.py (14%)
  - [ ] run_analysis.py — src/trend_analysis/run_analysis.py (13%)
  - [ ] market_data.py — src/trend_analysis/io/market_data.py (54%)
  - [ ] signal_presets.py — src/trend_analysis/signal_presets.py (52%)
  - [ ] frequency.py — src/trend_analysis/util/frequency.py (46%)
  - [ ] signals.py — src/trend_analysis/signals.py (65%)
  - [x] bootstrap.py — src/trend_analysis/backtesting/bootstrap.py (100%)
  - [ ] risk.py — src/trend_analysis/risk.py (71%)
  - [ ] bundle.py — src/trend_analysis/export/bundle.py (7%)
  - [ ] cli.py — src/trend_analysis/cli.py (10%)
  - [ ] optimizer.py — src/trend_analysis/engine/optimizer.py (23%)
  - [ ] model.py — src/trend_analysis/config/model.py (44%)
  - [ ] engine.py — src/trend_analysis/multi_period/engine.py (18%)

## Acceptance criteria
- [ ] Test coverage exceeds 95% for each file
- [ ] Essential functions for the program have full test coverage

## Coverage findings
- Soft coverage shows overall 42% line coverage with numerous high-priority gaps in the CLI, pipeline, presets, engine, and export subsystems. 【1aec47†L1-L90】
- Immediate focus areas include `pipeline.py`, `engine.py`, and `export/bundle.py`, which fall far below the 95% goal and represent critical runtime paths. 【1aec47†L34-L68】

## Next steps
- Prioritize targeted tests around `trend_analysis.pipeline`, `multi_period.engine`, and `export.bundle` to capture key execution paths and raise coverage towards the 95% target.
- Expand CLI- and configuration-focused suites (e.g., `trend_analysis.cli`, `config.model`, `io.validators`) to validate parameter parsing, error handling, and branching logic under diverse scenarios.
