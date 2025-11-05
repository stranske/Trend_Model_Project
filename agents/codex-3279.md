# Issue #3279 – Coverage Improvement Tracker

## Scope
Add test coverage for any program functionality with test coverage under 95% or for essential program functionality that does not currently have test coverage.

## Task Progress
- [x] Run soft coverage and prepare a list of the files with lowest coverage from least coverage on up for any file with less than 95% test coverage or any file with significant functionality that isn't covered.
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
  - [ ] frequency.py
  - [ ] signals.py
  - [ ] bootstrap.py
  - [ ] risk.py
  - [ ] bundle.py
  - [ ] cli.py
  - [ ] optimizer.py
  - [ ] model.py
  - [ ] engine.py

## Acceptance Criteria
- [ ] Test coverage exceeds 95% for each file.
- [ ] Essential functions for the program have full test coverage.

## Coverage Findings (2025-11-05)
Soft coverage was gathered with:

```bash
pytest --maxfail=1 --disable-warnings -k 'not test_autofix_pipeline_repairs_live_documents' --cov=src --cov-report=term
```

The failing live-document autofix scenario was excluded to let coverage complete; the scenario needs further work to restore parity with the recorded expectations.

Files below the 95% threshold, ordered from lowest coverage upward:

| File | Coverage |
| --- | --- |
| `src/health_summarize/__init__.py` | 0% |
| `src/trend_analysis/_autofix_probe.py` | 0% |
| `src/trend/reporting/unified.py` | 60% |
| `src/trend_analysis/io/market_data.py` | 91% |
| `src/trend_analysis/pipeline.py` | 91% |
| `src/trend_analysis/multi_period/engine.py` | 92% |
| `src/trend_analysis/proxy/cli.py` | 92% |
| `src/trend/cli.py` | 93% |
| `src/trend_analysis/cli.py` | 93% |
| `src/trend_analysis/export/bundle.py` | 93% |
| `src/trend_analysis/metrics/__init__.py` | 94% |
| `src/trend_analysis/config/legacy.py` | 95% |
| `src/trend_analysis/config/model.py` | 95% |
| `src/trend_analysis/engine/optimizer.py` | 95% |
| `src/trend_analysis/export/__init__.py` | 95% |
| `src/trend_analysis/metrics/rolling.py` | 95% |
| `src/trend_analysis/weights/hierarchical_risk_parity.py` | 95% |
| `src/trend_portfolio_app/app.py` | 95% |
| `src/trend_portfolio_app/monte_carlo/engine.py` | 95% |

Additional files already exceed 95% coverage and will be monitored while improvements proceed.

## Next Steps
1. Restore the live-document autofix regression scenario so coverage can be recorded without deselecting tests.
2. Prioritise coverage work starting with the zero-coverage modules, then progress through the remaining low-coverage files in order.
