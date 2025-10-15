# Self-Test Workflow Restriction Plan (Issue #2526)

## Scope and Key Constraints
- Limit automation to the GitHub Actions workflows `selftest-81-reusable-ci.yml` and `selftest-runner.yml` (the successors to the `selftest-8X-*` wrappers).
- Adjust only workflow trigger definitions and supporting documentation; do not change test logic or artefacts.
- Preserve the ability to run the self-tests manually through `workflow_dispatch` with minimal inputs (e.g., a free-form "reason").
- Avoid modifying unrelated workflows or repository automation beyond the targeted self-test files.
- Document updates must fit the existing CONTRIBUTING.md tone and structure while keeping instructions concise (two paragraphs).

## Acceptance Criteria / Definition of Done
1. The remaining self-test workflows (`selftest-81-reusable-ci.yml` and `selftest-runner.yml`) define **only** a `workflow_dispatch` trigger, optionally with a `reason` input, and have no other automatic triggers (`push`, `pull_request`, `schedule`, etc.). ✅ Verified for `selftest-81-reusable-ci.yml` and the runner.
2. No other workflows trigger self-tests implicitly; repository automation reflects the manual-only expectation. ✅ Guarded by `tests/test_workflow_selftest_consolidation.py`.
3. `CONTRIBUTING.md` contains a new two-paragraph section that explains the purpose of self-tests, when to run them, and how to interpret their output at a high level. ✅ Added in this iteration.
4. A self-test workflow has been manually triggered after the changes land, and the run link is recorded in PR communications. ⏳ Requires GitHub Actions UI access post-merge.

## Initial Task Checklist
- [x] Inventory all self-test workflow files under `.github/workflows/` (`selftest-81-reusable-ci.yml`, `selftest-runner.yml`) and confirm current triggers.
- [x] Update each targeted workflow to remove automatic triggers and keep (or add) a `workflow_dispatch` block with a `reason` input.
- [x] Validate workflow syntax locally with `./scripts/workflow_lint.sh` to ensure no YAML errors.
- [x] Draft and insert the new self-test guidance section into `CONTRIBUTING.md`, matching existing formatting conventions.
- [ ] Open a follow-up reminder to execute one manual self-test run once the workflows merge (or perform immediately if feasible) and share the link in the PR conversation.
