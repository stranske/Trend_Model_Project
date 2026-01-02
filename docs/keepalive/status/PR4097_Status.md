# Keepalive Status — PR #4097

> **Status:** In progress — clarifying `ci_level` behavior for reporting-only metadata.

## Progress updates
- Round 1: Reviewed current pipeline/config usage and confirmed `ci_level` is only surfaced as reporting metadata; added a regression test proving portfolio results are unchanged and documented the setting in config docs.

## Scope
The `ci_level` setting has no observable effect on portfolio construction.

## Tasks
- [x] Investigate intended use of `ci_level` setting
- [ ] If for construction: implement CI-based robust optimization (N/A: display-only)
- [x] If for display: document as reporting-only setting

## Acceptance criteria
- [x] Setting purpose is clearly documented
- [ ] If applicable: different CI levels produce different portfolios
