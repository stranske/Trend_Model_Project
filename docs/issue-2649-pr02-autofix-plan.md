# Issue #2649 — PR-02 Autofix Hardening Plan

> **2024 update.** The standalone `pr-02-autofix.yml` wrapper has been retired.
> Gate summary job Post CI now owns the opt-in autofix path and delegates directly to
> `reusable-18-autofix.yml`. This document remains as the historical plan for
> Issue #2649 with references updated to the Gate summary job entry point so maintainers
> can trace where each guarantee lives today.

## Scope / Key Constraints
- Limit automation to pull requests that explicitly opt in via the configurable `autofix:clean` label so routine PR events do not trigger unintended pushes; Gate summary job only calls the reusable job when the label is present and the loop guard permits a run.【F:.github/workflows/pr-00-gate.yml (summary job)†L808-L858】【F:.github/workflows/pr-00-gate.yml (summary job)†L1083-L1130】
- Keep the Gate summary job entry point as a thin wrapper around `reusable-18-autofix.yml`, ensuring all business logic flows through the reusable workflow for a single source of truth.【F:.github/workflows/pr-00-gate.yml (summary job)†L1083-L1177】【F:.github/workflows/reusable-18-autofix.yml†L1-L137】
- Enforce a shared `pr-${{ ... }}-gate` concurrency group so only one run per pull request can execute at a time and new events cancel superseded jobs.【F:.github/workflows/pr-00-gate.yml†L8-L20】【F:.github/workflows/reusable-18-autofix.yml†L95-L114】
- Preserve the existing permission set (`contents: write`, `pull-requests: write`) while making fork-safe decisions about how fixes are delivered to contributors.【F:.github/workflows/pr-00-gate.yml (summary job)†L22-L28】【F:.github/workflows/reusable-18-autofix.yml†L91-L214】
- Guarantee the reusable workflow emits a single consolidated status comment that links either the pushed commit or the uploaded patch artifact to avoid multi-comment noise.【F:.github/workflows/pr-00-gate.yml (summary job)†L1305-L1520】【F:.github/workflows/reusable-18-autofix.yml†L600-L775】

## Acceptance Criteria / Definition of Done
- For PRs lacking the opt-in label, Gate summary job evaluates the conditionals and exits without invoking the reusable job, leaving no automation traces or comments.【F:.github/workflows/pr-00-gate.yml (summary job)†L808-L858】【F:.github/workflows/pr-00-gate.yml (summary job)†L1083-L1148】
- Subsequent labelled/synchronise events on an opted-in PR cancel any in-flight run, and only the most recent execution completes.【F:.github/workflows/pr-00-gate.yml (summary job)†L8-L20】【F:.github/workflows/reusable-18-autofix.yml†L95-L114】
- Same-repo PRs with autofixable changes receive an automated commit, push, and comment linking the commit SHA; fork PRs receive a patch artifact, patch label, and instructions embedded in the comment.【F:.github/workflows/pr-00-gate.yml (summary job)†L1168-L1275】【F:.github/workflows/reusable-18-autofix.yml†L400-L608】
- Residual label hygiene (`autofix:clean` / `autofix:debt` / `autofix:patch`) and history artifacts continue to be maintained without regressing existing reporting behaviour.【F:.github/workflows/pr-00-gate.yml (summary job)†L1176-L1406】【F:.github/workflows/reusable-18-autofix.yml†L608-L775】
- Each automation run updates (or creates) exactly one status comment containing the autofix summary block and result details so reviewers have a consistent signal.【F:.github/workflows/pr-00-gate.yml (summary job)†L1305-L1520】【F:.github/workflows/reusable-18-autofix.yml†L600-L775】

## Initial Task Checklist
- [x] Audit label-gating logic for draft transitions and document manual label removal edge cases; Gate summary job exits when the opt-in label is absent and records loop-guard skips for future maintainers.【F:.github/workflows/pr-00-gate.yml (summary job)†L808-L858】【F:.github/workflows/pr-00-gate.yml (summary job)†L992-L1050】
- [x] Validate concurrency behaviour with back-to-back `synchronize` events so superseded runs cancel before side effects; both Gate summary job and the reusable workflow share the same per-PR group.【F:.github/workflows/pr-00-gate.yml (summary job)†L8-L20】【F:.github/workflows/reusable-18-autofix.yml†L95-L114】
- [x] Harden fork detection and patch publication with explicit commit staging, deterministic artifact naming, and application instructions for contributors without push access.【F:.github/workflows/pr-00-gate.yml (summary job)†L1257-L1520】【F:.github/workflows/reusable-18-autofix.yml†L400-L608】
- [x] Ensure the status comment builder consumes the `AUTOFIX_RESULT_BLOCK` in all paths, including no-change runs, so reviewers always see an Autofix result section.【F:.github/workflows/pr-00-gate.yml (summary job)†L1305-L1520】【F:scripts/build_autofix_pr_comment.py†L214-L270】
- [x] Reconfirm label hygiene automation obeys the new gating rules by scoping clean/debt updates to same-repo runs and clearing stale patch labels.【F:.github/workflows/pr-00-gate.yml (summary job)†L1176-L1406】
- [x] Update verification/runbook documentation with same-repo, fork, and gating scenarios to make manual validation straightforward post-change.【F:docs/autofix_type_hygiene.md†L82-L118】
