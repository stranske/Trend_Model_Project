# Keepalive Status — PR #4082

> **Status:** In progress — clarifying `ci_level` behavior.

## Progress updates
- Round 1: Reviewed recent commits and confirmed `ci_level` is only used for reporting UI/policy metadata; updated UI copy and wiring guard to reflect display-only intent.

## Scope
The `ci_level` setting has no observable effect on portfolio construction. It may only be used for display/reporting purposes.

## Tasks
- [x] Investigate intended use of `ci_level` setting
- [ ] If for construction: implement CI-based robust optimization (N/A: display-only)
- [x] If for display: document as reporting-only setting
- [x] Update tests/documentation accordingly

## Acceptance criteria
- [x] Setting's purpose is clearly documented
- [ ] If applicable: different CI levels produce different portfolios
- [x] If display-only: excluded from wiring validation tests
