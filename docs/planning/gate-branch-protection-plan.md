# Gate Branch Protection Implementation Plan

## Scope and Key Constraints
- **Target**: Default-branch protection must enforce the Gate GitHub Actions workflow (`.github/workflows/pr-gate.yml`) so every merge requires a passing Gate run.
- **Change Surface**: Branch-protection edits happen through GitHub UI/API by maintainers with admin permissions; in-repo work focuses on runbooks, validation evidence, and guardrail tooling rather than direct configuration.
- **Status-Check Hygiene**: Remove or archive any legacy required contexts (e.g., historical "CI" jobs) to prevent stale blockers once Gate is locked in.
- **Up-to-Date Requirement**: Enforce GitHub's "Require branches to be up to date before merging" toggle to guarantee Gate runs on the current tip of the default branch.
- **Documentation Impact**: Update contributor onboarding docs (notably `CONTRIBUTING.md`) without disrupting existing workflow guidance.
- **Validation Strategy**: Use a controlled draft PR (no unintended merges) to exercise both failing and passing Gate scenarios, capturing evidence for the issue record and repo-health automation.

## Acceptance Criteria / Definition of Done
1. The default branch rule lists Gate as a required status check **and** has "Require branches to be up to date" enabled.
2. No superseded required checks remain; the rule set only references workflows that still exist (Gate today).
3. `CONTRIBUTING.md` explicitly states that Gate must pass before merging to the default branch and references the up-to-date requirement.
4. A validation PR demonstrates the enforcement path: Gate blocks merge while red, then permits merge (or would permit, for draft) after the fix.
5. Runbook or planning docs (this file + validation notes) record the configuration steps and evidence so future audits can confirm compliance.

## Initial Task Checklist
1. **Inventory Current Protection Settings**
   - Capture the existing required-status list, the "up to date" toggle state, and any conditional rules.
   - Note whether automation helpers (e.g., `tools/enforce_gate_branch_protection.py`) already mirror the desired configuration.
2. **Apply Updated Rule**
   - In GitHub settings (or via the enforcement tool), enable required status checks, select Gate, and toggle "Require branches to be up to date".
   - Remove deprecated contexts so only live workflows remain.
3. **Refresh Documentation**
   - Update `CONTRIBUTING.md` (and linked onboarding snippets if needed) with the Gate requirement and the expectation to refresh stale branches.
4. **Validate with Draft PR Workflow**
   - Open a draft PR engineered to fail Gate, capture the merge-block UI, then push a fix to show Gate passing and the block clearing.
   - Store screenshots or run logs with the issue or repo-health evidence folder.
5. **Archive Findings and Tooling State**
   - Record the validation outcome, confirm scheduled enforcement (e.g., `health-44-gate-branch-protection.yml`) observes the new rule, and flag any follow-up automation gaps.

## Current Status
- ✅ **Documentation updated** – `CONTRIBUTING.md` now explains that the Gate check must pass before merging.
- ✅ **Automation helper** – `tools/enforce_gate_branch_protection.py` provides dry-run, check, and apply modes for owners with the
  appropriate token to configure branch protection without leaving the command line. The helper accepts explicit `--token` and
  `--api-url` overrides for GitHub Enterprise Server usage. Unit tests in `tests/tools/test_enforce_gate_branch_protection.py`
  cover the primary scenarios (initial bootstrap, drift detection, and clean enforcement). The CLI now supports `--snapshot` so
  enforcement runs can emit JSON evidence of the before/after branch protection state for audit logs.
- ✅ **Scheduled enforcement and verification** – `.github/workflows/health-44-gate-branch-protection.yml` now installs the helper,
  applies the rule when the optional `BRANCH_PROTECTION_TOKEN` secret is present, and then **always** runs `--check` with the
  default GitHub token. Drift immediately fails the workflow so repository owners are alerted even when the enforcement token is
  absent. Snapshot artifacts from each run are published for the evidence archive.
- 🔄 **Validation draft PR** – Pending repository-owner action. Follow the runbook in
  `docs/runbooks/gate-branch-protection-validation.md` to capture evidence of the Gate check blocking merges when failing and
  allowing merges when passing.
