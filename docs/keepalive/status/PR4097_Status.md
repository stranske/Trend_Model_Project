# Keepalive Status — PR #4097

> **Status:** Complete — clarified `ci_level` behavior for reporting-only metadata.

## Progress updates
- Round 1: Reviewed current pipeline/config usage and confirmed `ci_level` is only surfaced as reporting metadata; added a regression test proving portfolio results are unchanged and documented the setting in config docs.
- Round 2: Verified string `ci_level` inputs are coerced to reporting metadata and updated task checklist.

## Scope
The `ci_level` setting has no observable effect on portfolio construction.

## Tasks
- [x] Investigate intended use of `ci_level` setting
- [x] If for construction: implement CI-based robust optimization (N/A: display-only)
- [x] If for display: document as reporting-only setting

## Acceptance criteria
- [x] Setting purpose is clearly documented
- [x] If applicable: different CI levels produce different portfolios (N/A: reporting-only)
