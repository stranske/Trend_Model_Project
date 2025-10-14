# Branch Protection Enforcement Plan

## Scope and Key Constraints
- Confirm the repository's current default branch via the GitHub API and treat it as the single source of truth.
- Enforce the `Gate` status check as required **only** on the default branch; avoid modifying other branch rules.
- Ensure the Health-44 automation can run read-only when credentials are absent and enforce protection when a personal access token (PAT) is supplied.
- Maintain compatibility with existing workflow naming, secrets, and permissions; no new secrets should be introduced without documentation.
- Deliver documentation updates without breaking established references in `docs/ci/WORKFLOW_SYSTEM.md`.

## Acceptance Criteria / Definition of Done
- Health-44 workflow dynamically detects the default branch, validates that Gate is a required check, and reports "OK" when compliant.
- When a PAT with the required scopes is available, the workflow updates branch protection to require the Gate check if it is missing.
- A pull request targeting the default branch displays Gate as a required status check in the GitHub UI.
- `docs/ci/WORKFLOW_SYSTEM.md` describes the enforcement workflow, prerequisites, and audit steps for confirming branch protection.
- All workflow updates pass existing CI (Gate) without introducing regressions or unnecessary job executions.

## Initial Task Checklist
1. Retrieve and confirm the repository default branch programmatically (via the GitHub API) and locally (e.g., `git remote show origin`).
2. Audit current branch protection settings to determine whether Gate is already required.
3. Update `health-44-gate-branch-protection.yml` to:
   - Detect and export the default branch value once per run.
   - Validate Gate as a required check and optionally enforce it when a PAT is available.
   - Provide clear logging for both verification and enforcement paths.
4. Enhance `docs/ci/WORKFLOW_SYSTEM.md` with:
   - Step-by-step instructions for running the workflow manually.
   - Required permissions/secrets and expected outputs.
   - Guidance for auditing branch protection in the GitHub UI.
5. Run the Health-44 workflow (or simulate via `workflow_dispatch`) to confirm success and capture logs showing Gate marked as required.
6. Document verification results in the PR description and ensure reviewers can reproduce the checks.
