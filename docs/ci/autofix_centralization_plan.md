# Autofix Centralization Plan

## Scope and Key Constraints
- Consolidate all automated write operations that touch contributor pull requests into the `maint-46-post-ci` workflow, ensuring it remains the single authority for hygiene updates.
- Maintain fork safety: avoid force pushes to contributor branches when the PR source resides in a fork; instead, rely on downloadable patch artifacts delivered through workflow summaries and PR comments.
- Preserve existing loop guards and safety checks within the `maint-46-post-ci` workflow to prevent infinite retries or repeated patch uploads.
- For the pre-CI `pr-02-autofix` workflow, enforce opt-in behaviour via an `autofix` label and guarantee only one in-flight run per pull request by using GitHub Actions concurrency controls.
- Update documentation in `docs/ci/WORKFLOW_SYSTEM.md` to reflect the centralized autofix policy, fork-specific behaviour, and opt-in gating requirements.

## Acceptance Criteria / Definition of Done
1. `maint-46-post-ci.yml` remains the only workflow that writes to PR branches by default, with loop guards intact.
2. When a PR originates from a fork, Maint-46 publishes a patch artifact and surfaces a direct download link within the workflow summary or PR comment instead of pushing commits.
3. `pr-02-autofix.yml`, if retained, runs exclusively when the `autofix` label is attached to the PR and uses `cancel-in-progress` concurrency keyed by the PR number.
4. Workflow documentation clearly explains the centralized write policy, fork patch handling, and opt-in requirements for the pre-CI autofix job.
5. CI passes (workflow syntax validation via `act` dry-run or GitHub Actions) on the updated workflows.

## Initial Task Checklist
- [ ] Audit `maint-46-post-ci.yml` to confirm existing loop guards and identify fork-handling sections that need patch artifact uploads or messaging adjustments.
- [ ] Implement or verify patch artifact creation for fork-based runs and update the consolidated PR comment to link those artifacts.
- [ ] Modify `pr-02-autofix.yml` to add the `autofix` label gate and configure `concurrency: { group: pr-autofix-${{ github.event.pull_request.number }}, cancel-in-progress: true }`.
- [ ] Run linting or workflow validation (e.g., `act -n`) to ensure both workflows remain syntactically correct.
- [ ] Update `docs/ci/WORKFLOW_SYSTEM.md` with the new autofix policy and fork patch expectations.
- [ ] Capture before/after notes or screenshots of workflow summaries if needed for verification.
