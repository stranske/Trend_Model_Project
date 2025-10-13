# Health-40 Repo Self-Check Remediation Plan

## Scope and Key Constraints
- **Workflow rename**: Migrate `.github/workflows/repo-health-self-check.yml` to `.github/workflows/health-40-repo-selfcheck.yml` without breaking triggers or references.
- **Permissions compliance**: Limit workflow `permissions` to `contents: read`, `issues: write`, and `pull-requests: read`; all other scopes are out of scope.
- **Graceful privilege handling**: Constrain privileged GitHub API calls so they execute only when `SERVICE_BOT_PAT` is supplied. Without the PAT, the workflow must skip those checks without failing the job.
- **Reporting**: Ensure each check reports status in both the log output and `$GITHUB_STEP_SUMMARY` for observability during manual or scheduled runs.
- **Automation boundaries**: No changes to core analysis code or demo pipeline; work is confined to workflow YAML and the helper script invoked by the workflow.

## Acceptance Criteria / Definition of Done
1. Workflow file renamed to `.github/workflows/health-40-repo-selfcheck.yml`, and any internal or external references updated.
2. Workflow `permissions` block requests only supported scopes (`contents: read`, `issues: write`, `pull-requests: read`).
3. Workflow runs complete successfully on both `workflow_dispatch` and scheduled triggers without permission-related failures.
4. Privileged API calls are gated behind a check for `env.SERVICE_BOT_PAT`. When absent, the job surfaces a yellow “skipped privileged checks” note in the step summary while keeping the overall job successful.
5. `$GITHUB_STEP_SUMMARY` contains a concise Markdown table or bullet list summarizing each check and its outcome for every run.
6. A manual `workflow_dispatch` run is triggered, and the run URL is documented in the pull request discussion.

## Initial Task Checklist
- [ ] Rename the workflow file to `.github/workflows/health-40-repo-selfcheck.yml` and adjust references.
- [ ] Update the workflow `permissions` block to request only the supported scopes.
- [ ] Review the self-check script to identify privileged API calls; guard them behind a `SERVICE_BOT_PAT` presence check.
- [ ] Implement summary reporting that appends the outcome of each check to `$GITHUB_STEP_SUMMARY`.
- [ ] Validate workflow locally or via `act` (if feasible) to ensure logic paths succeed with and without `SERVICE_BOT_PAT`.
- [ ] Trigger a manual `workflow_dispatch` run in GitHub and comment the execution URL on the PR.
