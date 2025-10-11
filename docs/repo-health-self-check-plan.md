# Repo Health Workflow Remediation Plan

## Scope and Key Constraints
- Update `.github/workflows/repo-health-self-check.yml` so it validates under GitHub Actions, supports both `workflow_dispatch` and scheduled runs, and focuses strictly on repository health checks.
- Remove unsupported permission scopes (`metadata`, `administration`) and avoid introducing any other disallowed scopes; rely on default minimal permissions unless elevated access is essential for a specific API call.
- When privileged endpoints are required, use `secrets.SERVICE_BOT_PAT` without persisting it beyond the needed step, and ensure the workflow remains functional when the PAT is absent (read-only mode).
- Maintain the existing workflow cadence (at least daily or weekly) and keep runtime lightweight so it can run within standard GitHub-hosted runner limits.
- Provide actionable outputs via the job summary while avoiding noisy failures; the job should fail only on genuine health regressions rather than configuration issues.

## Acceptance Criteria / Definition of Done
- The workflow passes `act`/`workflow` validation and executes successfully via both the scheduled trigger and `workflow_dispatch`.
- Permissions configuration excludes unsupported scopes and aligns with GitHub Actions defaults; validation warnings about `metadata` or `administration` no longer appear.
- Steps that call privileged endpoints are wrapped in a condition that checks for the presence of `secrets.SERVICE_BOT_PAT`; those steps log a clear message when the secret is missing and gracefully skip privileged checks.
- The workflow emits a concise summary using the GitHub Actions step summary API, highlighting pass/fail status and pointing to any required remediation tasks.
- Automated health checks surface actionable failure reasons, and successful runs confirm repository health without raising false alarms.

## Initial Task Checklist
- [ ] Audit the existing `repo-health-self-check.yml` to document current jobs, steps, and failing permissions.
- [ ] Remove unsupported permission scopes and confirm remaining permissions satisfy required API calls.
- [ ] Isolate any step that needs elevated permissions; gate it behind an `if: env.SERVICE_BOT_PAT != ''` (or equivalent) check and wire the PAT through environment variables only within that step.
- [ ] Add logging that distinguishes between “PAT missing, skipped privileged checks” and actual errors to aid diagnostics.
- [ ] Implement or update a final step that writes an actionable summary to `$GITHUB_STEP_SUMMARY`, covering overall status and follow-up actions.
- [ ] Run the workflow via `workflow_dispatch` (and optionally `act`) to ensure it completes without permission errors and fails correctly on simulated regressions.
- [ ] Document any remaining follow-up work or open questions needed for full rollout.
