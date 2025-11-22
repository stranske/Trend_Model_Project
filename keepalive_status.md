# Keepalive Status for Upload Guard Enhancements

## Scope
- [ ] Use `st.cache_data` for expensive loads and derived tables; `st.cache_resource` if needed.
- [ ] Reject files above a configurable size (for example, 10 MB); sanitize headers starting with `=`, `+`, `-`, `@`.
- [ ] Validate schema before merging uploaded data with internal series; require unique column names.
- [ ] Extend `test_upload_app.py` with fuzzier inputs.

## Tasks
- [ ] Implement caching and guards.
- [ ] Add validation errors with clear messages in the UI.
- [ ] Extend tests for large files, duped columns, and formula-like headers.

## Acceptance criteria
- [ ] App accepts valid CSVs quickly on rerun and rejects malformed inputs with actionable messages.
- [ ] Tests cover the new edge cases.

Statuses remain unchecked until each acceptance criterion is met to keep the workflow nudging.
