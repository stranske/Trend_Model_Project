# Implementation Plan

## Workflow Audit
1. Inspect `.github/workflows/merge-manager.yml` to understand existing guardrails.
2. Confirm which jobs in `.github/workflows/pr-10-ci-python.yml` and `.github/workflows/pr-12-docker-smoke.yml` must be green before auto-merge unlocks.
3. Document how the Merge Manager bot determines whether a PR is documentation-only and whether it belongs to an allow-listed path set.

## Planned Changes
- Tighten the Merge Manager status checks so both CI and Docker workflows are required and reported explicitly in rationale messages.
- Require the `automerge` label before enabling auto-merge and log the absence as `missing automerge label`.
- Block any label containing `breaking`; surface the reason as `breaking label present` in the workflow output.
- Expand the unit tests under `tests/test_workflow_merge_manager.py` to cover the new status/label combinations and expected rationale text.

## Documentation & Task Tracking
- Update `agents/codex-1667.md` after workflow changes so the guardrails and validation checklist stay current.
- Mirror the acceptance criteria in `testing-plan.md` to guide manual verification of the merged automation.
- Record follow-up tasks for future tightenings (e.g., additional allow-list patterns) in `docs/ops/codex-bootstrap-facts.md` if scope increases.
