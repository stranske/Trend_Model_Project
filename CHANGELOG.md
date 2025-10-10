# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com/) and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- Regression tests (`test_shift_safe_pipeline.py`, `test_shift_safe_regression.py`) enforcing causal, no look‑ahead behavior in signal/position pipeline (Issue #1438).
- Centralised market-data validation enforcing a shared ingest contract across the CLI and Streamlit app, including cadence inference and return/price mode detection (Issue #1677).
- Bootstrap equity-band helper with Streamlit toggle and export bundle integration for uncertainty visualisation (Issue #1682).
- Reusable `verify-agent-assignment.yml` workflow that checks `agent:codex` issues for valid automation assignees and emits markdown summaries, available via manual dispatch or reusable calls (Issue #2386).

### Changed
- `compute_signal` now returns a strictly causal rolling mean shifted by one period (previously included the current row). Prevents subtle look‑ahead bias in downstream position construction (Issue #1438).

### Deprecated
- Legacy root scripts (`portfolio_analysis_report.py`, `manager_attribution_analysis.py`, `demo_turnover_cap.py`) now emit `DeprecationWarning` and delegate to unified `trend` CLI (Issue #1437).

### Migration Notes
- If external code depended on the previous inclusive rolling mean, apply `.shift(-1)` to approximate prior behavior, or compute the unshifted rolling mean directly via `df[column].rolling(window).mean()`.

---
