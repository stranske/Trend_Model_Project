# Keepalive Status for Issue #3770

## Scope
- [x] Introduce focused helpers for each stage (preprocessing, sampling/window resolution, selection, weighting/risk scaling, benchmarking/report assembly) and have `_run_analysis` orchestrate them.
- [x] Return structured result/diagnostic objects instead of bare `None` so callers can trace why a run produced no output.
- [x] Preserve existing behaviour by keeping the public API intact while internally delegating to the new helpers.

## Tasks
- [x] Identify logical sub-steps in `_run_analysis` and extract them into testable functions with clear inputs/outputs.
- [x] Add unit tests per helper covering success and failure paths.
- [x] Update `_run_analysis` to orchestrate helpers and propagate structured diagnostics.
- [x] Verify existing entry points (CLI/API) handle the richer return values without behaviour changes.

## Acceptance criteria
- [x] `_run_analysis` reads as a short orchestrator calling extracted helpers.
- [x] Unit tests cover the new helpers and confirm early exits report explicit reasons instead of returning `None`.
- [x] Existing external behaviour (inputs/outputs) remains unchanged aside from added diagnostics.

Status auto-updates as tasks complete on this branch.
