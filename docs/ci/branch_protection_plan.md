# Branch Protection Enforcement Plan

## Scope and Key Constraints
- Confirm the repository's current default branch via the GitHub API and treat it as the single source of truth.
- Enforce the `Gate` and `Health 45 Agents Guard` status checks as required on the default branch; avoid modifying other branch rules.
- Ensure the Health-44 automation can run read-only when credentials are absent and enforce protection when a personal access token (PAT) is supplied.
- Maintain compatibility with existing workflow naming, secrets, and permissions; no new secrets should be introduced without documentation.
- Deliver documentation updates without breaking established references in `docs/ci/WORKFLOW_SYSTEM.md`.

## Acceptance Criteria / Definition of Done
- Health-44 workflow dynamically detects the default branch, validates that Gate and the Health-45 Agents Guard are required checks, and reports "OK" when compliant.
- When a PAT with the required scopes is available, the workflow updates branch protection to require the Gate and Health-45 Agents Guard checks if either is missing.
- A pull request targeting the default branch displays Gate and the Health-45 Agents Guard as required status checks in the GitHub UI.
- `docs/ci/WORKFLOW_SYSTEM.md` describes the enforcement workflow, prerequisites, and audit steps for confirming branch protection.
- All workflow updates pass existing CI (Gate) without introducing regressions or unnecessary job executions.

## Initial Task Checklist
- [x] Retrieve and confirm the repository default branch programmatically (via the GitHub API) and locally (e.g., `git remote show origin`).
- `curl https://api.github.com/repos/stranske/Trend_Model_Project | python3 -c "import json,sys; print(json.load(sys.stdin)['default_branch'])"` → `phase-2-dev` on 2025-10-22.
- [x] Audit current branch protection settings to determine whether Gate is already required.
  - `curl https://api.github.com/repos/stranske/Trend_Model_Project/branches/phase-2-dev` reports `"Gate / gate"` and `"Health 45 Agents Guard / Enforce agents workflow protections"` under `required_status_checks.contexts` once branch protection is updated.
- [x] Update `health-44-gate-branch-protection.yml` to:
  - Detect and export the default branch value once per run.
  - Validate Gate as a required check and optionally enforce it when a PAT is available.
  - Provide clear logging for both verification and enforcement paths.
- [x] Enhance `docs/ci/WORKFLOW_SYSTEM.md` with:
  - Step-by-step instructions for running the workflow manually.
  - Required permissions/secrets and expected outputs.
  - Guidance for auditing branch protection in the GitHub UI.
- [ ] Run the Health-44 workflow (or simulate via `workflow_dispatch`) to confirm success and capture logs showing Gate and the Health-45 Agents Guard marked as required.
  - Pending manual dispatch with a PAT-backed `BRANCH_PROTECTION_TOKEN`; last scheduled runs failed while only Gate was enforced.
- [ ] Document verification results in the PR description and ensure reviewers can reproduce the checks.
  - To be finalised after a successful Health-44 execution produces updated artifacts/log URLs.

## Current Verification Evidence
- **Default branch**: `phase-2-dev` (GitHub REST API lookup on 2025-10-22).
- **Required status checks**: `Gate / gate` and `Health 45 Agents Guard / Enforce agents workflow protections` must appear under `required_status_checks.contexts` per `GET /repos/:owner/:repo/branches/:branch`.
- **Health-44 run status**: The two most recent scheduled runs (`18487152194`, `18456704713`) completed with `failure` while only Gate was enforced; re-run is required to capture an "OK" result once both checks are required.
