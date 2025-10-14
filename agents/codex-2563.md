# Issue #2563 — Self-test workflows manual-only plan

## Scope & Key Constraints
- Limit changes to self-test GitHub Actions workflows (`maint-43-selftest-pr-comment.yml`, `maint-44-*`, `maint-48-*`, `pr-20-selftest-pr-comment.yml`, and all `selftest-*` files) plus CONTRIBUTING documentation; avoid touching unrelated CI pipelines.
- Ensure every targeted workflow triggers exclusively through `workflow_dispatch` with the shared optional `reason` input; remove PR, push, and schedule triggers without altering job logic.
- Preserve existing job steps, secrets, and permissions; only update event triggers and shared inputs to prevent regressions.
- Confirm manual execution viability without relying on repository secrets unavailable to forks; document any prerequisites if discovered.

## Acceptance Criteria / Definition of Done
- All specified self-test workflows declare only the standardized `workflow_dispatch` trigger with the optional `reason` input (default `"manual test"`).
- No other trigger events (push, pull_request, schedule, etc.) remain in the affected workflow files.
- CONTRIBUTING.md includes a concise (3–5 sentence) section explaining the purpose of self-test workflows, how to run them manually via the `reason` input, and what successful runs validate.
- A manual dispatch of one updated self-test workflow completes successfully, with the run URL recorded in the PR description or follow-up comment.
- CI passes with no new linting or formatting issues introduced.

## Initial Task Checklist
- [ ] Inventory all self-test workflows in `.github/workflows/` and confirm the target list aligns with issue #2563.
- [ ] Update each workflow’s `on:` block to the standardized `workflow_dispatch` configuration while retaining existing jobs.
- [ ] Double-check for residual non-manual triggers (including `pull_request_target`, `workflow_call`, or reusable workflow references) and remove/adjust as needed.
- [ ] Add the new manual-execution guidance section to `CONTRIBUTING.md`, referencing the shared `reason` input.
- [ ] Trigger a representative self-test workflow manually and capture the successful run link for documentation.
- [ ] Review `git diff` for unintended changes and run any required formatting checks (e.g., `pre-commit`, `ruff`) if applicable.
- [ ] Summarize verification steps and the manual run link in the PR notes before requesting review.
