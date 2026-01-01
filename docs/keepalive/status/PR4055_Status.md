# Keepalive Status — PR #4055

> **Status:** In progress — regime detection wiring and behavior verification.

## Progress updates
- Round 1: Reviewed recent commits for regime wiring, added results-page coverage for regime proxy inputs, and validated targeted tests.
- Round 2: Added random-selection regime wiring test and ran `pytest -p no:rerunfailures tests/test_pipeline_optional_features.py -m "not slow"`.

## Scope
The regime detection settings (`regime_enabled`, `regime_proxy`) have no observable effect on portfolio construction. Users expect regime detection to modify fund selection or weights based on market conditions.

## Tasks
- [x] Trace `regime_enabled` and `regime_proxy` from UI through Config
- [x] Verify Config.regime settings are used in pipeline
- [x] If not used, implement regime-conditional logic in selection/weighting
- [x] Add wiring tests to verify settings have observable effect

## Acceptance criteria
- [x] Enabling `regime_enabled` changes portfolio behavior
- [x] Different `regime_proxy` values produce different results
- [x] Settings wiring tests pass for regime settings
- [x] No regression in existing tests
