<!-- bootstrap for codex on issue #3352 -->

# Coverage Improvement Initiative: Issue #3352

## Scope
Add Trend Analysis test coverage for any program functionality below the 95 % target or lacking regression safeguards, focusing on the modules enumerated by the keepalive checklist.

## Task Progress
- [x] Run soft coverage and prepare the ranked low-coverage list (`pytest tests/trend_analysis --cov=src/trend_analysis --cov-report=term-missing`).【ff80b1†L1-L86】
- [ ] Increase test coverage incrementally for one set of related issues or one file at a time
  - [ ] `src/trend_analysis/__init__.py` – 77 % (add import/metadata regression tests).【ff80b1†L15-L18】
  - [x] `src/trend_analysis/data.py` – 99 % (already above target; keep existing suite).【ff80b1†L32-L33】
  - [x] `src/trend_analysis/presets.py` – 97 % (already above target; monitor for regressions).【ff80b1†L56-L57】
  - [ ] `src/trend_analysis/backtesting/harness.py` – 14 % (needs scenario coverage).【ff80b1†L24-L27】
  - [ ] `src/trend_analysis/regimes.py` – 9 % (exercise regime detection branches).【ff80b1†L70-L71】
  - [ ] `src/trend_analysis/pipeline.py` – 7 % (cover orchestration happy/exception paths).【ff80b1†L48-L52】
  - [ ] `src/trend_analysis/io/validators.py` – 10 % (add validation success/failure tests).【ff80b1†L60-L61】
  - [ ] `src/trend_analysis/run_analysis.py` – 0 % (smoke test CLI integration wrapper).【ff80b1†L72-L73】
  - [ ] `src/trend_analysis/io/market_data.py` – 15 % (fixture-driven ingestion cases).【ff80b1†L58-L59】
  - [ ] `src/trend_analysis/signal_presets.py` – 0 % (cover registry wiring).【ff80b1†L74-L75】
  - [ ] `src/trend_analysis/util/frequency.py` – 20 % (frequency conversion edge cases).【ff80b1†L78-L79】
  - [ ] `src/trend_analysis/signals.py` – 37 % (expand indicator regression coverage).【ff80b1†L76-L77】
  - [ ] `src/trend_analysis/backtesting/bootstrap.py` – 11 % (bootstrap path coverage).【ff80b1†L21-L23】
  - [ ] `src/trend_analysis/risk.py` – 21 % (risk metric fan-out tests).【ff80b1†L68-L69】
  - [ ] `src/trend_analysis/export/bundle.py` – 7 % (export path scaffolding).【ff80b1†L44-L45】
  - [ ] `src/trend_analysis/cli.py` – 0 % (CLI smoke/integration tests).【ff80b1†L28-L29】
  - [ ] `src/trend_analysis/engine/optimizer.py` – 7 % (optimizer branch coverage).【ff80b1†L34-L35】
  - [ ] `src/trend_analysis/config/model.py` – 23 % (config validation cases).【ff80b1†L38-L41】
  - [ ] `src/trend_analysis/multi_period/engine.py` – 0 % (walk-forward scheduler coverage).【ff80b1†L64-L65】

## Acceptance Criteria
- [ ] Test coverage exceeds 95 % for each file
- [ ] Essential functions for the program have full test coverage

## Coverage Findings
The targeted Trend Analysis coverage run highlights the modules with the largest deltas below 95 %:

| Module | Coverage | Notes |
| --- | --- | --- |
| `src/trend_analysis/cli.py` | 0 % | No smoke or end-to-end tests exercise CLI entry points yet.【ff80b1†L28-L29】 |
| `src/trend_analysis/multi_period/engine.py` | 0 % | Walk-forward orchestration lacks regression coverage.【ff80b1†L64-L65】 |
| `src/trend_analysis/signal_presets.py` | 0 % | Registry helpers and preset wiring remain untested.【ff80b1†L74-L75】 |
| `src/trend_analysis/run_analysis.py` | 0 % | Script wrapper has no invocation smoke test.【ff80b1†L72-L73】 |
| `src/trend_analysis/export/bundle.py` | 7 % | Export bundler requires scenario coverage for success/error flows.【ff80b1†L44-L45】 |
| `src/trend_analysis/pipeline.py` | 7 % | Core pipeline orchestration is largely untested.【ff80b1†L48-L52】 |
| `src/trend_analysis/engine/optimizer.py` | 7 % | Optimizer code paths need deterministic fixtures.【ff80b1†L34-L35】 |
| `src/trend_analysis/regimes.py` | 9 % | Regime segmentation logic lacks data-driven tests.【ff80b1†L70-L71】 |
| `src/trend_analysis/io/validators.py` | 10 % | Validation layer has minimal assertions around expected schemas.【ff80b1†L60-L61】 |
| `src/trend_analysis/backtesting/bootstrap.py` | 11 % | Bootstrap routines need sampling/resampling tests.【ff80b1†L21-L23】 |
| `src/trend_analysis/backtesting/harness.py` | 14 % | Harness integration scenarios remain uncovered.【ff80b1†L24-L27】 |
| `src/trend_analysis/io/market_data.py` | 15 % | Market data ingestion lacks fixture coverage for edge cases.【ff80b1†L58-L59】 |
| `src/trend_analysis/util/frequency.py` | 20 % | Frequency helper requires conversion edge-case coverage.【ff80b1†L78-L79】 |
| `src/trend_analysis/risk.py` | 21 % | Risk aggregation paths need targeted verification.【ff80b1†L68-L69】 |
| `src/trend_analysis/config/model.py` | 23 % | Config model validation scenarios remain incomplete.【ff80b1†L38-L41】 |
| `src/trend_analysis/signals.py` | 37 % | Indicator generation should gain deterministic fixtures.【ff80b1†L76-L77】 |
| `src/trend_analysis/__init__.py` | 77 % | Additional import/metadata tests required to hit 95 %.【ff80b1†L15-L18】 |
| `src/trend_analysis/presets.py` | 97 % | Currently satisfies the acceptance target; monitor for regressions.【ff80b1†L56-L57】 |
| `src/trend_analysis/data.py` | 99 % | Currently satisfies the acceptance target; monitor for regressions.【ff80b1†L32-L33】 |

## Next Steps
1. Prioritise CLI, pipeline, and optimizer smoke/regression tests to lift 0–7 % modules above threshold.
2. Develop fixtures covering validators, market data ingestion, and frequency helpers to address mid-tier coverage gaps.
3. Extend integration-style tests for harness/regimes/multi-period engine components to capture complex orchestrations.
4. Re-run the targeted coverage suite after each module-focused improvement and refresh this log.
