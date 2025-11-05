<!-- bootstrap for codex on issue #3279 -->

# Coverage Improvement Initiative: Issue #3279

## Scope
Improve test coverage for the Trend Analysis project, prioritising `src/trend_analysis/` modules whose coverage remains below 95 % or whose behaviour is critical to the production pipeline.

## Task Progress
- [x] Run soft coverage (full-suite `coverage run -m pytest`) and prepare a ranked list of sub-95 % files (see "Coverage Findings").
- [ ] Increase test coverage incrementally for one related area at a time
  - [ ] `src/trend_analysis/__init__.py` – 85 % coverage; gaps in lazy loader error paths remain.【b9bb1e†L47-L47】
  - [ ] `src/trend_analysis/data.py` – 29 % coverage; loader branches and error handling untested.【b9bb1e†L94-L94】
  - [ ] `src/trend_analysis/presets.py` – 63 % coverage; dynamic preset expansion lacks assertions.【b9bb1e†L79-L79】
  - [ ] `src/trend_analysis/backtesting/harness.py` – 100 % coverage; no follow-up needed.【b9bb1e†L83-L83】
  - [ ] `src/trend_analysis/regimes.py` – 20 % coverage; regime derivation logic uncovered.【b9bb1e†L100-L100】
  - [ ] `src/trend_analysis/pipeline.py` – 57 % coverage; orchestration branches largely untested.【b9bb1e†L101-L101】
  - [ ] `src/trend_analysis/io/validators.py` – 14 % coverage; validation error cases need fixtures.【b9bb1e†L89-L89】
  - [ ] `src/trend_analysis/run_analysis.py` – 13 % coverage; CLI glue paths missing tests.【b9bb1e†L87-L87】
  - [ ] `src/trend_analysis/io/market_data.py` – 54 % coverage; ingestion error scenarios uncovered.【b9bb1e†L102-L102】
  - [ ] `src/trend_analysis/signal_presets.py` – 75 % coverage; interpolation branches uncovered.【b9bb1e†L72-L72】
  - [ ] `src/trend_analysis/util/frequency.py` – 46 % coverage; sparsely-sampled frequency helpers untested.【b9bb1e†L86-L86】
  - [ ] `src/trend_analysis/signals.py` – 65 % coverage; signal factory fallbacks need exercising.【b9bb1e†L75-L75】
  - [ ] `src/trend_analysis/backtesting/bootstrap.py` – 100 % coverage; no follow-up needed.【b9bb1e†L82-L82】
  - [ ] `src/trend_analysis/risk.py` – 71 % coverage; risk window guards remain uncovered.【b9bb1e†L85-L85】
  - [ ] `src/trend_analysis/export/bundle.py` – 7 % coverage; exporter wiring lacks test doubles.【b9bb1e†L91-L91】
  - [ ] `src/trend_analysis/cli.py` – 87 % coverage; option dispatch paths missing assertions.【b9bb1e†L93-L93】
  - [ ] `src/trend_analysis/engine/optimizer.py` – 20 % coverage; solver branching uncovered.【b9bb1e†L95-L95】
  - [ ] `src/trend_analysis/config/model.py` – 44 % coverage; config migration logic uncovered.【b9bb1e†L96-L96】
  - [ ] `src/trend_analysis/multi_period/engine.py` – 18 % coverage; walk-forward scheduling untested.【b9bb1e†L98-L98】

## Acceptance Criteria
- [ ] Test coverage exceeds 95 % for each file listed above.
- [ ] Essential functions for the program have full test coverage.

## Coverage Findings
Coverage was gathered with `python -m coverage run -m pytest` followed by `python -m coverage report -m | sort -k4,4n`. The run failed at `tests/test_autofix_pipeline_live_docs.py::test_autofix_pipeline_repairs_live_documents`, but the generated report still identifies the lowest-coverage targets for follow-up.【9dd456†L1-L137】【b9bb1e†L1-L111】 Representative sub-95 % files and their current coverage levels:

| Module | Coverage | Notes |
| --- | --- | --- |
| `src/trend_analysis/export/bundle.py` | 7 % | Vast majority of exporter wiring lacks isolation or integration tests.【b9bb1e†L91-L91】 |
| `src/trend_analysis/multi_period/engine.py` | 18 % | Walk-forward evaluation and failure paths remain untested.【b9bb1e†L98-L98】 |
| `src/trend_analysis/io/validators.py` | 14 % | Input schema failures and CSV error handling need fixtures.【b9bb1e†L89-L89】 |
| `src/trend_analysis/pipeline.py` | 57 % | Only core happy-path logic runs in tests; branch coverage is sparse.【b9bb1e†L101-L101】 |
| `src/trend_analysis/data.py` | 29 % | Data loading fallbacks and error branches are missing coverage.【b9bb1e†L94-L94】 |
| `src/trend_analysis/cli.py` | 87 % | CLI option permutations (signals, export modes) remain untested.【b9bb1e†L93-L93】 |

See the full `coverage report` output for additional modules requiring attention.【b9bb1e†L1-L111】

## Next Steps
1. Stabilise `tests/test_autofix_pipeline_live_docs.py::test_autofix_pipeline_repairs_live_documents` in a scratch branch so the full-suite coverage run completes without manual interruption.【9dd456†L1-L137】
2. Design fixture-driven unit tests for `src/trend_analysis/data.py` and `src/trend_analysis/io/market_data.py` to lift loader coverage above 95 %.
3. Exercise success and failure paths in `src/trend_analysis/pipeline.py`, ideally by orchestrating stubbed components.
4. Expand CLI smoke tests to capture export and signal toggles in `src/trend_analysis/cli.py`.
5. Re-run targeted coverage after each module-focused test addition and update this log accordingly.

*Last updated: 2025-02-15*
