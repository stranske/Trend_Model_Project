# Keepalive Status — PR #4065

> **Status:** Complete — trend signal settings wired into analysis output.

## Progress updates
- Round 1: Confirmed signal settings are threaded into pipeline and updated tests to validate window/z-score effects.

## Scope
The `config.signals` configuration (containing `trend_window` and `trend_zscore` settings) is constructed but never passed to `_run_analysis` in the API layer. Users can configure these settings in the UI but they have **no effect** on actual analysis output.

## Tasks
- [x] Identify where signal computation occurs in the pipeline (likely in data preprocessing or trend calculation)
- [x] Add `signals_cfg` parameter to `_run_analysis()` function signature in `src/trend_analysis/api.py`
- [x] Pass `signals_cfg` through the pipeline to the signal generation code
- [x] Use `trend_window` and `trend_zscore` values from config in the actual calculations
- [x] Add unit test verifying that changing `trend_window` produces different output
- [x] Add unit test verifying that changing `trend_zscore` produces different output

## Acceptance criteria
- [x] Running analysis with `trend_window=20` produces different results than `trend_window=60`
- [x] Running analysis with `trend_zscore=1.0` produces different results than `trend_zscore=2.0`
- [x] All existing tests continue to pass
- [x] No regressions in pipeline behavior
