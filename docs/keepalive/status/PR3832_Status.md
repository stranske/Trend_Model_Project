# Keepalive Status — PR #3832

> **Status:** In progress — address Issue #3820 by validating index membership and surfacing clear feedback.

## Progress updates
- Round 1: Captured the bootstrap scope, tasks, and acceptance criteria for issue #3820 and registered the checklist in the keepalive index.
- Round 2: Implemented index membership validation, surfaced requested/accepted/missing benchmark feedback, and covered the new flows with tests.

## Scope
- [x] Validate requested indices against available window data before selection proceeds.
- [x] Report which indices were accepted and which were ignored or missing so misconfigurations are obvious.
- [x] Cover index validation paths with tests for valid, partially missing, and fully missing lists.

## Tasks
- [x] Guard `_select_universe` against index columns that are absent or empty in the analysis window and fail fast when none are usable.
- [x] Thread requested/accepted/missing index details into selection metadata so downstream consumers see what was honored.
- [x] Add unit tests for fully valid, partially missing, and fully missing index lists (success and failure cases).

## Acceptance criteria
- [x] Universe selection fails gracefully with clear diagnostics when requested indices are absent from the window data.
- [x] Valid indices are selected without altering current behaviour; missing entries are reported deterministically in metadata.
- [x] Tests exercise mixed-validity scenarios and assert the reported messages.
