# Keepalive Status for Ranking Cleanup PR

## Scope
- [ ] Identify unused ranking utilities and remove them or move them into a clearly marked legacy/experimental module.
- [ ] Ensure public APIs and tests reference only the supported ranking functions.
- [ ] Add or update tests to confirm the active ranking path remains intact after cleanup.

## Tasks
- [ ] Trace usage of ranking utilities and delete or isolate unused items with appropriate documentation.
- [ ] Update imports/call sites and adjust tests accordingly.
- [ ] Add/maintain tests to ensure the supported ranking functions operate as before.

## Acceptance criteria
- [ ] `core/rank_selection.py` contains only actively used utilities (or clearly segregated legacy stubs).
- [ ] Test suite passes with the cleaned-up module and no stray references to removed utilities.

Status auto-updates as tasks complete on this branch.
