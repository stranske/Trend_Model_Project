# Issue #2733 Gate Docs-Only Comment Retirement – Planning Notes

## Scope and Key Constraints
- **Docs-only behaviour**: Confine changes to the Gate workflow (and companion tests/docs) that govern docs-only detection and reporting. Avoid introducing new workflows or altering unrelated automation.
- **PR conversation hygiene**: Eliminate the brittle docs-only PR comment while ensuring contributors still get a visible signal via logs and the job summary.
- **Idempotency**: Clean up any legacy docs-only comments so repeated runs stay stable and do not leave duplicate markers behind.
- **Documentation parity**: Update the existing CI documentation set rather than creating ad-hoc guides; keep terminology aligned with prior Gate docs.

## Acceptance Criteria / Definition of Done
1. Docs-only runs publish their fast-pass outcome exclusively through logs and the step summary; no PR comment is created. ✅
2. Legacy docs-only comments (matching the historical body or marker) are removed automatically on every run. ✅
3. Targeted automation tests assert the docs-only handler behaviour and guard the cleanup logic. ✅
4. CI documentation (README + workflow references) describes the docs-only summary path and cleanup. ✅
5. Focused pytest modules covering Gate automation succeed locally. ✅

## Task Checklist
- [x] Update `pr-00-gate.yml` to remove the comment writer, extend the docs-only summary, and harden the cleanup step.
- [x] Strengthen workflow tests to cover the comment-free handler and legacy cleanup detection.
- [x] Refresh README / CI docs to point out the summary-only behaviour and automatic cleanup.
- [x] Capture the plan, acceptance criteria, and completion evidence here.
- [x] Run `pytest tests/test_automation_workflows.py tests/test_workflow_naming.py -q`.
