# Keepalive Status for Issue #3770 `_run_analysis` Restructure

## Scope
- [ ] Introduce focused helpers for each stage (preprocessing, sampling/window resolution, selection, weighting/risk scaling, benchmarking/report assembly) and have `_run_analysis` orchestrate them.
- [ ] Return structured result/diagnostic objects instead of bare `None` so callers can trace why a run produced no output.
- [ ] Preserve existing behaviour by keeping the public API intact while internally delegating to the new helpers.

## Tasks
- [ ] Identify logical sub-steps in `_run_analysis` and extract them into testable functions with clear inputs/outputs.
- [ ] Add unit tests per helper covering success and failure paths.
- [ ] Update `_run_analysis` to orchestrate helpers and propagate structured diagnostics.
- [ ] Verify existing entry points (CLI/API) handle the richer return values without behaviour changes.

## Acceptance criteria
- [ ] `_run_analysis` reads as a short orchestrator calling extracted helpers.
- [ ] Unit tests cover the new helpers and confirm early exits report explicit reasons instead of returning `None`.
- [ ] Existing external behaviour (inputs/outputs) remains unchanged aside from added diagnostics.

Status auto-updates as tasks complete on this branch.
