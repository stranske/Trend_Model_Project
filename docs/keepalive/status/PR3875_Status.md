# Keepalive Status — PR #3875

> **Status:** In progress — recorded the scope, tasks, and acceptance criteria so keepalive continues nudging until completion.

## Progress updates
- Round 1: Captured the initial scope/tasks/acceptance from the automated status summary for issue #3861 (risk-free fallback window coverage and diagnostics).

## Scope
- [ ] The risk-free fallback chooses the lowest-volatility numeric column with any non-null value, so sparse series can be selected and later propagate NaNs when aligned to the analysis windows.

## Tasks
- [ ] Enforce minimum coverage within the requested window before considering a column eligible for fallback selection (e.g., threshold on non-null share or span).
- [ ] Align fallback volatility checks to the same window slice used downstream rather than the full dataset.
- [ ] Add tests that include sparse columns and confirm only sufficiently covered, window-aligned series are selected (or a clear diagnostic is emitted).

## Acceptance criteria
- [ ] - Fallback risk-free selection ignores sparse or short series and uses window-scoped data for volatility comparison.
- [ ] - Omitted risk-free column scenarios produce deterministic, well-documented outcomes with no unexpected NaNs from missing coverage.
- [ ] - Tests cover explicit risk-free columns, eligible fallback selections, and rejection of sparse series with corresponding diagnostics.
