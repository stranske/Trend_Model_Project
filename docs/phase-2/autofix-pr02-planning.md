# Autofix PR-02 Hardening Plan

## Scope and Key Constraints
- **Workflow coverage**: Changes limited to the `pr-02-autofix` entry workflow and the reusable Autofix workflow it invokes. Other CI workflows stay untouched to avoid unintended concurrency coupling.
- **Concurrency model**: Use GitHub Actions `concurrency` groups keyed to the pull request head ref so multiple pushes serialize instead of overlap. Must allow reruns on different branches to proceed simultaneously.
- **Label gating**: Autofix should run only when an explicit label (e.g., `autofix`) is present. On missing label, workflow must exit cleanly without side effects.
- **Fork safety**: Forked pull requests run with read-only permissions; workflow must not attempt privileged artifact uploads or write-scoped operations when running from forks.
- **Comment hygiene**: Bot feedback consolidated into a single updatable comment per pull request to prevent notification spam.
- **Secrets handling**: Ensure no secret values or write tokens leak via artifacts, logs, or uploads; prefer minimal permission scopes in workflow `permissions` block.

## Acceptance Criteria / Definition of Done
- Concurrent pushes to the same pull request head ref never produce overlapping Autofix jobs; rerunning cancels the prior in-progress run.
- Autofix workflow exits early without modification attempts when the required label is absent.
- Forked pull request runs complete without attempting to upload privileged artifacts and do not error due to insufficient permissions.
- The bot leaves at most one status comment per pull request and edits it on retries instead of posting new messages.
- Documentation updated where necessary to reflect new gating, concurrency, and fork-safety behavior.
- All updated workflows lint/validate successfully (e.g., `act --list` or `yamllint` as applicable) and pass required CI checks.

## Initial Task Checklist
- [ ] Audit existing Autofix workflows to identify where concurrency, permissions, and labeling logic should live.
- [ ] Implement or update the `concurrency` block to scope runs by head ref and enable `cancel-in-progress`.
- [ ] Add label detection step that short-circuits execution (with informative log) when the label is missing.
- [ ] Tighten workflow `permissions` to the minimum required; add conditional guards around artifact or comment steps for forked repositories.
- [ ] Refactor bot commenting logic to reuse a single comment, updating it on reruns.
- [ ] Update documentation (e.g., README or workflow docs) to describe the new operational expectations.
- [ ] Run or simulate workflows to confirm no regressions and that forked PRs behave safely.
