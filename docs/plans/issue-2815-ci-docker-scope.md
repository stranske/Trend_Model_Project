# CI Docker Scope Plan (Issue #2815)

## Scope and Key Constraints
- Limit Docker smoke testing to pull requests that modify Docker-related assets while preserving existing fast-pass behavior for documentation-only changes.
- Detect modifications across `Dockerfile`, `.dockerignore`, `.docker/`, and `docker/` paths using a deterministic changed-files mechanism that integrates with the Gate workflow.
- Ensure compatibility with existing workflow outputs and any reusable jobs that rely on the Gate workflow's `detect` step outputs.
- Maintain current reporting expectations in the Gate summary so that skipped Docker jobs surface clearly without breaking downstream automation.

## Acceptance Criteria / Definition of Done
- Gate workflow surfaces a boolean-like output (e.g., `docker_changed`) that correctly reflects whether any Docker-scoped files were touched in the PR.
- Docker smoke job executes only when `docker_changed == 'true'` while continuing to honor doc-only fast pass behavior for other jobs.
- Skipped Docker smoke runs still appear in workflow summaries with an explicit "skipped" status and informative messaging.
- Documentation or inline comments explain the changed-files filter logic for future maintenance.
- CI run on a PR without Docker changes skips the Docker smoke job; a PR with Docker changes triggers the job.

## Initial Task Checklist
1. ✅ Update the Gate workflow to compute and expose `docker_changed` using a native Git diff step.
2. ✅ Propagate the new output to relevant job-level conditions so the Docker smoke job keys off `steps.detect.outputs.docker_changed`.
3. ✅ Verify that doc-only detection remains intact and compatible with the new Docker gating logic (no changes needed after review).
4. ✅ Adjust Gate summary/reporting scripts so they surface the skipped Docker job status without errors.
5. ✅ Exercise the workflow logic with representative change sets (Docker change vs. non-Docker change) via reasoning; CI verification will occur on subsequent runs.

## Acceptance Criteria Status
- ✅ PRs without Docker changes skip the Docker job by virtue of the `docker_changed` guard.
- ✅ PRs that touch Docker paths still run the smoke test when the guard evaluates to `true`.
- ✅ Workflow summary messaging now documents when Docker smoke is skipped due to unchanged Docker assets.
