# Self-Test Workflow Restriction Plan (Issue #2526)

> **2026-11-15 update (Issue #2728):** the consolidated self-test refresh restored
> the nightly cron on `selftest-runner.yml` once the reusable wrapper was retired.
> This document now serves as historical background for the manual-only window
> introduced by Issue #2526. The acceptance criteria below capture that temporary
> posture and remain satisfied for that effort, while the current configuration is
> described in [`docs/ci/WORKFLOWS.md`](ci/WORKFLOWS.md) and
> [`docs/ci/WORKFLOW_SYSTEM.md`](ci/WORKFLOW_SYSTEM.md).

## Scope and Key Constraints
- Limit automation to the GitHub Actions workflow `selftest-runner.yml` (the successor to the `selftest-8X-*` wrappers).
- Adjust only workflow trigger definitions and supporting documentation; do not change test logic or artefacts.
- Preserve the ability to run the self-tests manually through `workflow_dispatch` with minimal inputs (e.g., a free-form "reason").
- Avoid modifying unrelated workflows or repository automation beyond the targeted self-test files.
- Document updates must fit the existing CONTRIBUTING.md tone and structure while keeping instructions concise (two paragraphs).

## Acceptance Criteria / Definition of Done
1. The remaining self-test workflow (`selftest-runner.yml`) defines **only** a `workflow_dispatch` trigger, optionally with a `reason` input, and has no other automatic triggers (`push`, `pull_request`, `schedule`, etc.). ✅ Verified for the runner during the manual-only phase (superseded when the nightly cron returned in Issue #2728).
2. No other workflows trigger self-tests implicitly; repository automation reflects the manual-only expectation. ✅ Guarded by `tests/test_workflow_selftest_consolidation.py` during the restriction window (restored schedule now validated by the same guardrails).
3. `CONTRIBUTING.md` contains a new two-paragraph section that explains the purpose of self-tests, when to run them, and how to interpret their output at a high level. ✅ Added in this iteration.
4. A self-test workflow has been manually triggered after the changes land, and the run link is recorded in PR communications. ⏳ Requires GitHub Actions UI access post-merge.

## Initial Task Checklist
- [x] Inventory the self-test workflow file under `.github/workflows/` (`selftest-runner.yml`) and confirm current triggers.
- [x] Update the targeted workflow to remove automatic triggers and keep (or add) a `workflow_dispatch` block with a `reason` input.
- [x] Validate workflow syntax locally with `./scripts/workflow_lint.sh` to ensure no YAML errors.
- [x] Draft and insert the new self-test guidance section into `CONTRIBUTING.md`, matching existing formatting conventions.
- [ ] Open a follow-up reminder to execute one manual self-test run once the workflows merge (or perform immediately if feasible) and share the link in the PR conversation.
