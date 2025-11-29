# Keepalive Status — PR #3875

> **Status:** In progress — recorded the scope, tasks, and acceptance criteria so keepalive continues nudging until completion.

## Progress updates
- Round 1: Captured the initial scope/tasks/acceptance from the automated status summary for issue #3861 (risk-free fallback window coverage and diagnostics).
- Round 2: Reconfirmed all items remain open; no acceptance criteria have been satisfied yet. Checklists reposted for visibility until progress is made.
- Round 3: Per latest keepalive prompt, reiterated the current scope, tasks, and acceptance criteria; no task or acceptance item has been met yet, so all checklists remain open for continued nudging.
- Round 4: Implemented window-scoped coverage gating for fallback risk-free selection, aligned volatility checks to the combined in/out analysis window, and added sparse-coverage regression tests. All scope, task, and acceptance items are now satisfied.

## Scope
- [x] The risk-free fallback chooses the lowest-volatility numeric column with any non-null value, so sparse series can be selected and later propagate NaNs when aligned to the analysis windows.

## Tasks
- [x] Enforce minimum coverage within the requested window before considering a column eligible for fallback selection (e.g., threshold on non-null share or span).
- [x] Align fallback volatility checks to the same window slice used downstream rather than the full dataset.
- [x] Add tests that include sparse columns and confirm only sufficiently covered, window-aligned series are selected (or a clear diagnostic is emitted).

## Acceptance criteria
- [x] - Fallback risk-free selection ignores sparse or short series and uses window-scoped data for volatility comparison.
- [x] - Omitted risk-free column scenarios produce deterministic, well-documented outcomes with no unexpected NaNs from missing coverage.
- [x] - Tests cover explicit risk-free columns, eligible fallback selections, and rejection of sparse series with corresponding diagnostics.
