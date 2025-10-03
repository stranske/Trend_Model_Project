# Automerge Trigger Documentation ([Issue #1667](https://github.com/stranske/Trend_Model_Project/issues/1667))

## Purpose
This document outlines the acceptance criteria and implementation details for automerge triggers as described in Issue #1667. Automerge should occur only when all required checks pass and specific labeling conditions are met.

## Preconditions for Automerge
- **CI status is green** (all required checks have passed)
- **Docker build status is green** (Docker-related checks have passed)
- **PR has the `automerge` label**
- **PR does _not_ have any label containing `breaking`**

## Task List
- [ ] Audit automerge and supporting workflows to confirm they enforce the label and status rules from Issue #1667.
- [ ] Update automerge workflow logic so it requires successful CI and Docker runs for the PR head commit.
- [ ] Require the `automerge` label before auto-merge can be enabled.
- [ ] Block auto-merge when any label containing `breaking` is applied.
- [ ] Add or update tests/automation checks that cover the refined trigger conditions.
- [ ] Verify a docs-only PR with the `automerge` label merges automatically after all checks pass.
- [ ] Verify that PRs missing the `automerge` label or carrying a `breaking` label do **not** auto-merge.

## Implementation Notes
- **Relevant workflow files:**
  - `.github/workflows/pr-10-ci-python.yml` (primary CI jobs)
  - `.github/workflows/pr-12-docker-smoke.yml` (Docker build/test checks)
  - `.github/workflows/merge-manager.yml` (automerge orchestration)
- **Labels involved:**
  - `automerge` (required for merge automation)
  - Any label containing `breaking` (must be absent)

## Validation Checklist
- [ ] CI checks are green
- [ ] Docker checks are green
- [ ] `automerge` label is present
- [ ] No `breaking` label is present
- [ ] Automerge workflow triggers only when all above are true
