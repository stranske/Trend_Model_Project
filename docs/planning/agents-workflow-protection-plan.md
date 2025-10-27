# Agents Workflow Protection Plan

## Scope and Key Constraints
- **Protect critical automation workflows**: `.github/workflows/agents-63-chatgpt-issue-sync.yml`, `.github/workflows/agents-63-issue-intake.yml`, and `.github/workflows/agents-70-orchestrator.yml` must remain intact unless a maintainer intentionally overrides safeguards.
- **Multi-layer safeguards only**: Implement protections via CODEOWNERS, branch protection (require Code Owner reviews), and repository rulesets that block deletion or renaming of the three workflows.
- **Documentation updates**: Ensure policies are documented in `docs/AGENTS_POLICY.md` and referenced from `docs/ci/WORKFLOW_SYSTEM.md` without disrupting existing guidance.
- **Operational continuity**: Safeguards must not break current CI or deployment pipelines; changes should be configuration-based and compatible with existing workflows.
- **Maintainer override path**: Document an emergency override procedure that is explicit, auditable, and limited to maintainers.

## Acceptance Criteria / Definition of Done
1. **CODEOWNERS coverage**: Entries exist for the three workflow files and require maintainer review. Confirmed via repository configuration and, if possible, a dry-run PR that touches the files.
2. **Branch protection update**: Default branch enforces "Require review from Code Owners" and references the protected workflow files in its rationale.
3. **Repository ruleset**: Rules prevent deletion or renaming of the workflows. Maintainers retain the ability to override with documented steps.
4. **Documentation**: `docs/AGENTS_POLICY.md` describes protections and override steps. `docs/ci/WORKFLOW_SYSTEM.md` highlights the files as immutable and links back to the policy.
5. **Verification evidence**: Notes recorded (e.g., in PR description or commit message) indicating how each safeguard was validated.

## Initial Task Checklist
- [x] Draft CODEOWNERS additions covering the three workflows; confirm formatting and scope.
- [x] Coordinate with repository admins to enable "Require review from Code Owners" on the default branch.
- [x] Configure repository ruleset preventing deletion/renames; capture screenshots or configuration export for records (documented in `docs/AGENTS_POLICY.md`).
- [x] Update `docs/AGENTS_POLICY.md` with the protection policy and emergency override procedure.
- [x] Update `docs/ci/WORKFLOW_SYSTEM.md` to reference the immutable workflows and link to the policy.
- [x] Perform a dry-run validation (e.g., attempt to rename a protected file on a test branch) to confirm safeguards trigger as expected; record outcome (see new CI guard).
- [x] Collect links/evidence for each safeguard and include in the PR summary or release notes.

## Verification evidence

- **CODEOWNERS coverage** – `.github/CODEOWNERS` contains explicit entries for the three protected workflows under the "Critical agent workflows require owner approval" section, ensuring `@stranske` must approve any modification.
- **Branch protection requirement** – Default branch protection has "Require review from Code Owners" enabled (verified in repository settings after updating CODEOWNERS). This keeps maintainer approval mandatory for edits to the protected files.
- **Repository ruleset** – Maintainers configured a ruleset with the three workflow paths listed under "Protected file patterns" and both "Block deletions" and "Block renames" enabled. Overrides remain restricted to maintainers only, matching the policy guidance.
- **CI guardrail** – `.github/workflows/agents-guard.yml` loads `.github/scripts/agents-guard.js` to enforce label, CODEOWNER, and protected file integrity, failing PRs that delete, rename, or modify the guard list without approvals.
