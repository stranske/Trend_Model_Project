# Issue #1666 – CI signature guard refresh

## Task checklist

### Discovery
- [x] Review consolidated CI/self-test workflows to capture the current required job manifest (`.github/workflows/pr-00-gate.yml`, `.github/workflows/maint-90-selftest.yml`).
- [x] Confirm hashed fields and guard flow by inspecting `.github/workflows/maint-40-ci-signature-guard.yml` and `tools/test_failure_signature.py`.

### Fixture refresh
- [x] Ensure `.github/signature-fixtures/basic_jobs.json` enumerates every guarded job with updated step placeholders and stack tokens.
- [x] Regenerate `.github/signature-fixtures/basic_hash.txt` via `python tools/test_failure_signature.py --jobs "$(cat .github/signature-fixtures/basic_jobs.json)"`.
- [ ] Adjust guard workflow or helper script if fixture structure changes.

### Validation
- [x] Sanity-check refreshed fixtures with `python tools/test_failure_signature.py --jobs "$(cat .github/signature-fixtures/basic_jobs.json)" --expected "$(cat .github/signature-fixtures/basic_hash.txt)"`.
- [x] Perturb a job entry locally to verify hash mismatch detection still fires.
- [ ] Monitor the next "CI Signature Guard" workflow run for a green result with updated fixtures (post-merge follow-up).

