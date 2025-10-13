# Self-Test Workflow Restriction Plan (Issue #2526)

## Scope and Key Constraints
- Limit automation to the GitHub Actions workflows matching `selftest-8X-*`.
- Adjust only workflow trigger definitions and supporting documentation; do not change test logic or artefacts.
- Preserve the ability to run the self-tests manually through `workflow_dispatch` with minimal inputs (e.g., a free-form "reason").
- Avoid modifying unrelated workflows or repository automation beyond the targeted self-test files.
- Document updates must fit the existing CONTRIBUTING.md tone and structure while keeping instructions concise (two paragraphs).

## Acceptance Criteria / Definition of Done
1. Every `selftest-8X-*` workflow defines **only** a `workflow_dispatch` trigger, optionally with a `reason` input, and has no other automatic triggers (`push`, `pull_request`, `schedule`, etc.).
2. No other workflows trigger self-tests implicitly; repository automation reflects the manual-only expectation.
3. `CONTRIBUTING.md` contains a new two-paragraph section that explains the purpose of self-tests, when to run them, and how to interpret their output at a high level.
4. A self-test workflow has been manually triggered after the changes land, and the run link is recorded in PR communications.

## Initial Task Checklist
- [ ] Inventory all `selftest-8X-*` workflow files under `.github/workflows/` and confirm current triggers.
- [ ] Update each targeted workflow to remove automatic triggers and keep (or add) a `workflow_dispatch` block with a `reason` input.
- [ ] Validate workflow syntax locally (e.g., `act` dry run) or via GitHub's workflow editor to ensure no YAML errors.
- [ ] Draft and insert the new self-test guidance section into `CONTRIBUTING.md`, matching existing formatting conventions.
- [ ] Open a follow-up reminder to execute one manual self-test run once the workflows merge (or perform immediately if feasible) and share the link in the PR conversation.
