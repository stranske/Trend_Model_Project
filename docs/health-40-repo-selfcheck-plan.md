# Health-40 Repo Self-Check Remediation Plan

## Scope and Key Constraints
- **Workflow slug**: Ensure `.github/workflows/health-40-repo-selfcheck.yml` remains the active repo self-check (formerly `repo-health-self-check.yml`) without breaking triggers or references.
- **Permissions compliance**: Limit workflow `permissions` to `contents: read` and `issues: write`; all other scopes are out of scope.
- **Graceful privilege handling**: Constrain privileged GitHub API calls so they execute only when `SERVICE_BOT_PAT` is supplied. Without the PAT, the workflow must downgrade branch-protection visibility gaps—including 403/429 rate-limit responses—to warnings instead of failing the job.
- **Reporting**: Ensure each check reports status in both the log output and `$GITHUB_STEP_SUMMARY` for observability during manual or scheduled runs.
- **Automation boundaries**: No changes to core analysis code or demo pipeline; work is confined to workflow YAML and the helper script invoked by the workflow.

## Acceptance Criteria / Definition of Done
1. Workflow file renamed to `.github/workflows/health-40-repo-selfcheck.yml`, and any internal or external references updated.
2. Workflow `permissions` block requests only supported scopes (`contents: read`, `issues: write`).
3. Workflow runs complete successfully on both `workflow_dispatch` and scheduled triggers without permission-related failures.
4. Privileged API calls are gated behind a check for `env.SERVICE_BOT_PAT`. When absent, the job surfaces a yellow warning about downgraded branch protection visibility while keeping the overall job successful.
5. `$GITHUB_STEP_SUMMARY` contains a concise Markdown table or bullet list summarizing each check and its outcome for every run.
6. A manual `workflow_dispatch` run is triggered, and the run URL is documented in the pull request discussion.

## Initial Task Checklist
- [x] Rename the workflow file to `.github/workflows/health-40-repo-selfcheck.yml` and adjust references.
- [x] Update the workflow `permissions` block to request only the supported scopes.
- [x] Review the self-check script to identify privileged API calls; guard them behind a `SERVICE_BOT_PAT` presence check.
- [x] Implement summary reporting that appends the outcome of each check to `$GITHUB_STEP_SUMMARY`.
- [ ] Validate workflow locally or via `act` (if feasible) to ensure logic paths succeed with and without `SERVICE_BOT_PAT`.
- [ ] Trigger a manual `workflow_dispatch` run in GitHub and comment the execution URL on the PR.

### Validation Notes

- 2025-10-13: Attempted to execute the workflow locally using `act`, but the container runtime is unavailable in the current environment (`Cannot connect to the Docker daemon`). Re-run the validation from an environment with Docker access to complete this checklist item.
- Manual `workflow_dispatch` remains outstanding; once a GitHub runner triggers the job, capture the run URL in the PR discussion to close the final acceptance criterion.
- 2025-10-13: Verified that unauthenticated calls to `POST /actions/workflows/health-40-repo-selfcheck.yml/dispatches` return `401` (requires authentication). Prepare a PAT-backed invocation, for example:

  ```bash
  curl -X POST \
       -H "Authorization: Bearer ${SERVICE_BOT_PAT}" \
       -H "Accept: application/vnd.github+json" \
       https://api.github.com/repos/stranske/Trend_Model_Project/actions/workflows/health-40-repo-selfcheck.yml/dispatches \
       -d '{"ref": "work"}'
  ```

  Replace `SERVICE_BOT_PAT` with an appropriately scoped token before rerunning so the manual dispatch succeeds and the resulting run URL can be recorded in this PR.
