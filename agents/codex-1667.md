# Automerge Trigger Documentation ([Issue #1667](https://github.com/stranske/Trend_Model_Project/issues/1667))

## Purpose
This document outlines the acceptance criteria and implementation details for automerge triggers as described in Issue #1667. Automerge should occur only when all required checks pass and specific labeling conditions are met.

> **Update (2026-10-12):** Issue #2190 removed `.github/workflows/maint-45-merge-manager.yml` and the associated labeler workflow. The checklist below is retained for historical context; the repository no longer ships an automerge workflow by default.

## Preconditions for Automerge
- **CI status is green** (the Gate workflow's Python jobs have passed)
- **Docker build status is green** (the Gate workflow's Docker smoke job has succeeded)
- **PR has the `automerge` label** (explicit opt-in for minor-change auto-merges)
- **PR does _not_ have any label containing `breaking`** (guards against high-risk changes)

### Additional Guardrails Enforced by Merge Manager
- `ci:green` label is synchronized with the CI run and must be present before auto-merge
- Allow-listed file patterns only (docs/tests/lightweight config)
- PR must carry a `risk:low` label and an agent attribution label (`from:codex` / `from:copilot`)
- Quiet period elapsed and no active CI/Autofix workflow executions

## Task List
- [x] Audit automerge and supporting workflows to confirm they enforce the label and status rules from Issue #1667 (`.github/workflows/maint-45-merge-manager.yml`).
- [x] Update automerge workflow logic so it requires successful CI and Docker runs for the PR head commit (see "Check required workflows" gate).
- [x] Require the `automerge` label before auto-merge can be enabled (reason logged as "missing automerge label").
- [x] Block auto-merge when any label containing `breaking` is applied (reason logged as "breaking label present").
- [x] Add or update tests/automation checks that cover the refined trigger conditions (`tests/test_workflow_merge_manager.py`). (Historical; removed alongside the workflow.)
- [x] Verify a docs-only PR with the `automerge` label merges automatically after all checks pass (simulated via Merge Manager logic inspection and unit coverage).
- [x] Verify that PRs missing the `automerge` label or carrying a `breaking` label do **not** auto-merge (covered by guarded reasons and tests).

## Implementation Notes
- **Relevant workflow files:**
  - `.github/workflows/pr-gate.yml` (primary CI/Docker orchestration)
  - `.github/workflows/maint-45-merge-manager.yml` (automerge orchestration; removed in Issue #2190)
- **Labels involved:**
  - `automerge` (required for merge automation)
  - Any label containing `breaking` (must be absent)

## Validation Checklist
- [ ] CI checks are green
- [ ] Docker checks are green
- [ ] `automerge` label is present
- [ ] No `breaking` label is present
- [ ] Automerge workflow triggers only when all above are true
