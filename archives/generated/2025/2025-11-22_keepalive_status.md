# Keepalive Status for Issue #3771 Cleanup PR

## Scope
- [x] Identify unused ranking utilities and remove them or move them into a clearly marked legacy/experimental module.
- [x] Ensure public APIs and tests reference only the supported ranking functions.
- [x] Add or update tests to confirm the active ranking path remains intact after cleanup.

## Tasks
- [x] Trace usage of ranking utilities and delete or isolate unused items with appropriate documentation.
- [x] Update imports/call sites and adjust tests accordingly.
- [x] Add/maintain tests to ensure the supported ranking functions operate as before.

## Acceptance criteria
- [x] `core/rank_selection.py` contains only actively used utilities (or clearly segregated legacy stubs).
- [x] Test suite passes with the cleaned-up module and no stray references to removed utilities.

Status auto-updates as tasks complete on this branch.
