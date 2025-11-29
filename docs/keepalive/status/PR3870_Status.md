# Keepalive Status — PR #3870

> **Status:** In progress — registered the keepalive checklist for issue #3860 so tasks can be worked through with window-aligned risk-free fallback updates.

## Progress updates
- Round 1: Captured the PR's current scope, tasks, and acceptance criteria; no implementation work has started yet.

## Scope
- [ ] Scope section missing from source issue.

## Tasks
- [ ] Enforce minimum coverage within the requested window before considering a column eligible for fallback selection (e.g., threshold on non-null share or span).
- [ ] Align fallback volatility checks to the same window slice used downstream rather than the full dataset.
- [ ] Add tests that include sparse columns and confirm only sufficiently covered, window-aligned series are selected (or a clear diagnostic is emitted).

## Acceptance criteria
- [ ] Fallback risk-free selection ignores sparse or short series and uses window-scoped data for volatility comparison.
- [ ] Omitted risk-free column scenarios produce deterministic, well-documented outcomes with no unexpected NaNs from missing coverage.
- [ ] Tests cover explicit risk-free columns, eligible fallback selections, and rejection of sparse series with corresponding diagnostics.

## Implementation notes
- [ ] Target `_resolve_risk_free_column` (and related fallback helpers) to apply coverage thresholds and window scoping without changing metric formulas.
- [ ] Synced by [workflow run](https://github.com/stranske/Trend_Model_Project/actions/runs/19750667918).
