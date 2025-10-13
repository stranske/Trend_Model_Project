# Issue #2527 — Gate Branch Protection Enforcement Plan

## Scope & Constraints
- Apply and verify branch protection on the default branch so the **Gate / gate** status check is required before merges.
- Keep the enforcement automated through `tools/enforce_gate_branch_protection.py` and `.github/workflows/health-44-gate-branch-protection.yml`; avoid manual UI-only changes without recording evidence.
- Preserve any intentional legacy alias (for example a single informational `CI` context) only when Gate remains required and strict mode (`require branches to be up to date`) is enabled.
- Coordinate with repository administrators for tokens that grant `Administration → Branches` scope; do not embed credentials in the repository.

## Acceptance Criteria / Definition of Done
1. Default branch protection rule lists **Gate / gate** in the required status checks and has `strict: true`.
2. Optional aliases (if any) remain informational; Gate continues to be the blocking requirement.
3. Evidence bundle under `docs/evidence/gate-branch-protection/` contains snapshots from the helper script (pre-check, enforcement if needed, and post-check) plus validation PR proof.
4. The scheduled `health-44-gate-branch-protection` workflow succeeds, demonstrating automated verification of the rule.

## Task Checklist
- [ ] Confirm the current default branch (`gh repo view --json defaultBranchRef`).
- [ ] Run `tools/enforce_gate_branch_protection.py --check --snapshot docs/evidence/gate-branch-protection/pre-enforcement.json` with an admin-scoped token.
- [ ] If the helper reports drift, rerun with `--apply` and capture `enforcement.json`.
- [ ] Trigger or wait for `health-44-gate-branch-protection` to confirm the configuration (store `post-enforcement.json`).
- [ ] Open a validation pull request from a non-default branch, cancel or fail Gate, capture merge blocking evidence, then rerun Gate to collect the success state.
- [ ] Update `agents/codex-2527.md` once evidence is archived and the workflow is healthy.

## Validation Notes
- Follow `docs/runbooks/gate-branch-protection-validation.md` for the validation PR procedure and evidence expectations.
- Ensure snapshots redact secrets and include timestamps produced by the helper (`generated_at` field).
- Archive workflow artifacts locally and commit the JSON summaries to the evidence folder for auditability.

## Resources
- Helper script: [`tools/enforce_gate_branch_protection.py`](../../tools/enforce_gate_branch_protection.py)
- Automation workflow: [`.github/workflows/health-44-gate-branch-protection.yml`](../../.github/workflows/health-44-gate-branch-protection.yml)
- Prior plan: [`docs/planning/gate-branch-protection-plan.md`](./gate-branch-protection-plan.md)
- Runbook: [`docs/runbooks/gate-branch-protection-validation.md`](../runbooks/gate-branch-protection-validation.md)
