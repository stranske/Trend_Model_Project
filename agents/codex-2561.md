<!-- bootstrap for codex on issue #2561 -->

# Scope & Key Constraints

- Limit changes to `.github/workflows/health-40-repo-selfcheck.yml` and ancillary documentation or automation notes directly supporting the repo health self-check workflow.
- Preserve the existing schedule (`cron: '20 6 * * 1'`) and `workflow_dispatch` triggers; do not introduce additional automatic triggers or remove manual invocation.
- Maintain read-only operational posture except where the workflow already requires `issues: write`; avoid elevating permissions beyond the validated set (`contents: read`, `pull-requests: read`, `issues: write`, `actions: read`).
- Ensure optional use of `secrets.SERVICE_BOT_PAT`; the workflow must continue to execute successfully when the secret is absent.
- Avoid modifications that would break compatibility with current GitHub-hosted runner environments (Ubuntu latest) or require new repository secrets.

# Acceptance Criteria / Definition of Done

1. The workflow passes GitHub Actions validation and runs successfully via both `workflow_dispatch` and the scheduled trigger.
2. Branch protection probing is performed only when `SERVICE_BOT_PAT` is supplied; when absent, the step emits a notice and records a neutral/"skipped" result without failing the job.
3. All privileged API calls are guarded behind the optional PAT and emit actionable warnings/notices when access is forbidden or skipped.
4. The job appends a Markdown summary table to `$GITHUB_STEP_SUMMARY`, capturing each health check's title, status, and details, plus an annotation when privileged checks were skipped.
5. Updated permissions exclude the unsupported `administration` scope while retaining required read/write scopes, and the workflow documentation (PR summary) reflects these adjustments.

# Initial Task Checklist

- [x] Audit the existing workflow to confirm current triggers, permissions, and branch protection logic.
- [x] Update the permissions block to remove `administration` and add any required supported scopes (e.g., `actions: read`).
- [x] Ensure the branch-protection step consumes `SERVICE_BOT_PAT` when available (via `github-token` or environment) while degrading gracefully when the secret is absent.
- [x] Implement or refine the summary-writing step to output a Markdown table to `$GITHUB_STEP_SUMMARY`, including skipped privileged-check messaging.
- [ ] Manually trigger the workflow (or plan for maintainers to do so) and capture run metadata/link for validation. _(Pending maintainer-run; workflow dispatch is available but cannot be triggered from this environment.)_
