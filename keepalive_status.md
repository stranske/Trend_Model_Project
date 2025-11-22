# Keepalive Status for Upload Guard Enhancements

## Scope
- [x] Use `st.cache_data` for expensive loads and derived tables; `st.cache_resource` if needed.
- [x] Reject files above a configurable size (for example, 10 MB); sanitize headers starting with `=`, `+`, `-`, `@`.
- [x] Validate schema before merging uploaded data with internal series; require unique column names.
- [x] Extend `test_upload_app.py` with fuzzier inputs.

## Tasks
- [x] Implement caching and guards.
- [x] Add validation errors with clear messages in the UI.
- [x] Extend tests for large files, duped columns, and formula-like headers.

## Acceptance criteria
- [x] App accepts valid CSVs quickly on rerun and rejects malformed inputs with actionable messages.
- [x] Tests cover the new edge cases.

All checklist items are complete per the satisfied acceptance criteria; future updates should reset items only if new scope emerges.
