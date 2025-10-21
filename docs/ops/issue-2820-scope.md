# Issue #2820 – Clean-mode cosmetic autofix

_Last reviewed: 2025-10-19_

## Scope & Key Constraints
- Introduce a label-gated `autofix:clean` mode that reuses the existing cosmetic autofix workflow while constraining edits to `tests/**`.
- Keep the standard `autofix` behaviour unchanged and allow maintainers to opt into the scoped mode by applying the additional label.
- Reuse Ruff import sorting, lint fixes, and formatting without widening rule coverage beyond the current workflow defaults.
- Maintain present automation ergonomics (status comments, label management, artifact handling) and ensure runs abort if non-test paths are touched.
- Preserve green Gate status after the new mode is exercised.

## Acceptance Criteria / Definition of Done
1. [x] A demo PR with the `autofix:clean` label shows automation modifying only files beneath `tests/**`.
2. [x] The workflow performs import reordering, whitespace normalisation, and simple Ruff fixes exclusively within the scoped files.
3. [x] Automation posts a succinct tests-only summary comment enumerating the modified files when changes are applied.
4. [x] Gate completes successfully after the changes.
5. [x] Documentation and status surfaces reflect how to trigger and interpret the tests-only mode.

_All acceptance criteria met._

## Task Checklist
- [x] Review `.github/workflows/maint-46-post-ci.yml` and `.github/workflows/reusable-18-autofix.yml` to wire the `autofix:clean` label through trigger and guard logic.
- [x] Restrict Ruff invocations to `tests/` when the tests-only mode is active and abort if the diff contains non-test files.
- [x] Extend `scripts/build_autofix_pr_comment.py` and regression tests to highlight the tests-only mode in the consolidated status comment.
- [x] Document request/verification steps for the scoped sweep in contributor hygiene and CI workflow references.
- [x] Capture a focused pytest run covering the updated comment builder logic.

## Verification Log
- 2025-10-19: `pytest tests/test_build_autofix_pr_comment.py` (pass) to confirm comment reporting for the tests-only mode.
- 2025-10-19: Manual inspection of `.github/workflows/reusable-18-autofix.yml` ensures Ruff sweeps abort if non-`tests/` files change and records affected test files for reviewer summary comments.
