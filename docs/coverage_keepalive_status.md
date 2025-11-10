# Coverage Status — Keepalive Round

## Soft coverage snapshot

- Generated a soft coverage report using `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m coverage run --source=src -m pytest` followed by `python -m coverage report -m`.
- Captured the ranked module list from `coverage_report.txt`; the lowest coverage files (<50%) currently are:
  1. `src/trend_analysis/export/__init__.py` — 7 %
  2. `src/trend_analysis/export/bundle.py` — 7 %
  3. `src/trend_analysis/engine/walkforward.py` — 14 %
  4. `src/trend_analysis/core/rank_selection.py` — 17 %
  5. `src/trend_analysis/config/model.py` — 44 %
  6. `src/trend_analysis/config/models.py` — 46 %
  7. `src/trend_analysis/data.py` — 29 %
  8. `src/trend_analysis/cli.py` — 10 %
  9. `src/trend_analysis/regimes.py` — 19 %
  10. `src/trend_analysis/core/metric_cache.py` — 36 %

(See `coverage_report.txt` for the full listing.)

## Test execution blockers

- Re-running the focused CLI regression suite against Python 3.12 / NumPy 2.1 failed with `_NoValueType` conversion errors raised by NumPy’s ufuncs when pandas resampling relies on deprecated behaviour. The failure log is stored at `tmp_logs/run_analysis_fail.log` for reference.
- Similar `_NoValueType` failures surface when running the backtesting harness and frequency utilities test packs, so targeted coverage improvements for those modules are blocked until we patch the NumPy compatibility issues in the fixtures.

## Next steps

1. Patch the pandas/Numpy interaction in the CLI, frequency, and backtesting fixtures so the suites run cleanly on Python 3.12/NumPy 2.1.
2. Once the regression suites pass, re-run the soft coverage command and lift the remaining modules above the 95 % target starting with `export/__init__.py`, `engine/walkforward.py`, and `config/model.py`.
3. Document the updated coverage deltas in `docs/coverage_progress.md` after confirming the fixes.
