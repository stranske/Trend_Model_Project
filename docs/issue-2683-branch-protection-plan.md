# Issue 2683 – Gate + Agents Guard Branch Protection Plan

## Scope and Key Constraints
- Update the default-branch protection rule so **both** `Gate / gate` and `Health 45 Agents Guard` status checks are required simultaneously; no other contexts should be removed unless explicitly deprecated.
- Preserve the existing `require up to date` setting and any additional protections (code owner reviews, signed commits, etc.) while modifying the required checks list.
- Extend `tools/enforce_gate_branch_protection.py` (or successor helper) to treat both contexts as first-class requirements, supporting dry-run and enforcement modes without broadening the token scope beyond `Administration: Branches`.
- Keep the Health 44 verification workflow lightweight: it must succeed with only the default token when checks are correct and fail fast when either required context is missing or renamed.
- Avoid duplicating configuration across scripts and workflows; rely on a single source of truth (e.g., helper defaults or a shared constant) so future changes are centralized.
- Update contributor-facing documentation (CONTRIBUTING.md, docs/ci/WORKFLOW_SYSTEM.md) without altering unrelated guidance or historical notes.

## Acceptance Criteria / Definition of Done
1. Default branch protection lists exactly `Gate / gate` and `Health 45 Agents Guard` as required checks and keeps the `strict` flag enabled.
2. `tools/enforce_gate_branch_protection.py` (or replacement) enforces and verifies both required checks, emitting clear diff output when drift is detected.
3. `.github/workflows/health-44-gate-branch-protection.yml` fails when either required check is missing, renamed, or replaced and passes when both are present.
4. Documentation enumerates the two required checks, explains their purpose, and references the enforcement helper plus Health 44 guard for remediation.
5. Evidence (workflow logs or helper snapshots) demonstrates the enforcement path and the failure mode when a check is intentionally removed.

## Initial Task Checklist
- [ ] Audit current branch protection settings (via UI or API) to capture the existing required checks and `strict` flag state.
- [ ] Refactor or extend the enforcement helper so the desired contexts list includes both `Gate / gate` and `Health 45 Agents Guard`, with reusable constants for downstream consumers.
- [ ] Update the Health 44 workflow to assert the dual-context requirement and surface actionable failure messaging for missing checks.
- [ ] Regenerate or adjust unit tests covering the enforcement helper to account for the new context set and snapshot outputs.
- [ ] Refresh contributor and workflow documentation to highlight the dual required checks and outline recovery steps when drift occurs.
- [ ] Run the enforcement helper in dry-run and apply modes (as permissions allow) to record before/after states and confirm branch protection matches the desired configuration.
- [ ] Trigger the Health 44 workflow (manually or via PR) to validate that it passes with correct settings and fails when a required check is removed.
