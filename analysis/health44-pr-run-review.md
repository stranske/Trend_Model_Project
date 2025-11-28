# Health 44 PR Run Hang Review

## Summary
- Queried the latest PR executions of the **Health 44 Gate Branch Protection** workflow via the public Actions API. Only run **19750914472** (PR #3822 on `codex/issue-3817`) remains `in_progress`; the preceding four PR runs finished successfully.【83bdcb†L1-L6】【1e1f45†L1-L4】
- Job-step metadata shows the hanging run stopped in the **"Verify gate branch protection is required (admin scope)"** step. Enforcement was skipped (as expected for PR events), so the admin-scope verification is the first step that actually exercises `tools/enforce_gate_branch_protection.py` and it has not completed; the read-only fallback and artifact upload never started.【b1622b†L1-L15】【0440d2†L137-L199】
- Completed PR runs from the same window finished the admin-scope verification instantly and moved on to restoring snapshots and uploading artifacts, suggesting the hang is specific to the current run, not systemic in the workflow definition.【800d49†L1-L17】
- Attempting to download the raw logs from GitHub returned a `403` (“Must have admin rights to Repository”), so deeper step output was unavailable from this environment.【b0cd1a†L1-L5】

## Likely cause
The workflow runs the admin-scope verification even on `pull_request` events when `BRANCH_PROTECTION_TOKEN` is present (see workflow lines 137–176). In this run, that step never returned, implying the call to `tools/enforce_gate_branch_protection.py --check --require-strict` is blocked—likely due to the admin token hitting rate-limit/permissions issues or a stalled API response. Because the job never reaches the read-only verification or summary steps, the overall check appears to run indefinitely.【0440d2†L137-L199】【b1622b†L1-L15】

## Recommendations
1. **Add an explicit timeout around the admin-scope verification step** (e.g., `timeout-minutes: 5` on the step or job) so PR runs fail fast instead of hanging when the admin token cannot complete the API call.【0440d2†L137-L199】
2. **Fallback to the read-only verification when the admin step times out or errors on PRs.** That keeps the PR signal unblocked while still surfacing drift via the standard snapshot artifact.
3. **Audit the `BRANCH_PROTECTION_TOKEN` usage on PRs** (rate limits, scope, or fork restrictions). If it is not needed for PR verification, adjust the workflow condition to skip the admin path for PR events and rely on the read-only check instead.【0440d2†L137-L176】
