# Issue #2649 — PR-02 Autofix Hardening Plan

## Scope / Key Constraints
- Limit automation to pull requests that explicitly opt in via the configurable `autofix` label so routine PR events do not trigger unintended pushes.【F:.github/workflows/pr-02-autofix.yml†L3-L31】
- Keep the PR-02 entrypoint as a thin wrapper around `reusable-18-autofix.yml`, ensuring all business logic flows through the reusable workflow for a single source of truth.【F:.github/workflows/pr-02-autofix.yml†L20-L31】【F:.github/workflows/reusable-18-autofix.yml†L1-L123】
- Enforce a shared `pr-02-autofix-${{ github.event.pull_request.number || github.run_id }}` concurrency group so only one run per pull request can execute at a time and new events cancel superseded jobs.【F:.github/workflows/pr-02-autofix.yml†L16-L18】【F:.github/workflows/reusable-18-autofix.yml†L21-L33】
- Preserve the existing permission set (`contents: write`, `pull-requests: write`) while making fork-safe decisions about how fixes are delivered to contributors.【F:.github/workflows/reusable-18-autofix.yml†L17-L210】
- Guarantee the reusable workflow emits a single consolidated status comment that links either the pushed commit or the uploaded patch artifact to avoid multi-comment noise.【F:.github/workflows/reusable-18-autofix.yml†L179-L383】

## Acceptance Criteria / Definition of Done
- For PRs lacking the opt-in label, PR-02 evaluates the conditional and exits without invoking the reusable job, leaving no automation traces or comments.【F:.github/workflows/pr-02-autofix.yml†L20-L31】
- Subsequent labeled/synchronize events on an opted-in PR cancel any in-flight run, and only the most recent execution completes.【F:.github/workflows/pr-02-autofix.yml†L16-L18】【F:.github/workflows/reusable-18-autofix.yml†L21-L33】
- Same-repo PRs with autofixable changes receive an automated commit, push, and comment linking the commit SHA; fork PRs receive a patch artifact, patch label, and instructions embedded in the comment.【F:.github/workflows/reusable-18-autofix.yml†L123-L260】
- Residual label hygiene (`autofix:clean` / `autofix:debt` / `autofix:patch`) and history artifacts continue to be maintained without regressing existing reporting behaviour.【F:.github/workflows/reusable-18-autofix.yml†L264-L416】
- Each automation run updates (or creates) exactly one status comment containing the autofix summary block and result details so reviewers have a consistent signal.【F:.github/workflows/reusable-18-autofix.yml†L358-L383】

## Initial Task Checklist
1. Audit label-gating logic for draft PR transitions and document any edge cases requiring additional safeguards (e.g., manual label removal paths).【F:.github/workflows/pr-02-autofix.yml†L20-L31】
2. Validate concurrency behaviour with back-to-back `synchronize` events and confirm cancelled runs stop before side effects (commit/push/comment).【F:.github/workflows/pr-02-autofix.yml†L16-L18】【F:.github/workflows/reusable-18-autofix.yml†L21-L33】
3. Harden the fork detection and patch publication steps, covering retries and artifact naming conventions for GitHub UI clarity.【F:.github/workflows/reusable-18-autofix.yml†L197-L260】
4. Ensure the status comment builder consumes the `AUTOFIX_RESULT_BLOCK` for both commit and patch flows and gracefully handles runs with no changes.【F:.github/workflows/reusable-18-autofix.yml†L179-L383】
5. Reconfirm label hygiene automation interacts correctly with the new gating rules, especially when an autofix run exits early or produces no diffs.【F:.github/workflows/reusable-18-autofix.yml†L264-L383】
6. Update verification/runbook documentation with testing steps for same-repo and fork scenarios once the workflow changes are in place.【F:.github/workflows/reusable-18-autofix.yml†L123-L383】
