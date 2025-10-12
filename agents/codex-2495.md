<!-- Codex tracking for issue #2495: Enforce PR branch protection on Gate -->

## Status Snapshot
- [x] Documented scope, acceptance criteria, and checklist (`docs/planning/gate-branch-protection-plan.md`).
- [x] CONTRIBUTING guide updated with Gate enforcement requirement.
- [x] Automation script and nightly workflow available to audit/enforce branch protection.
- [ ] Validation draft PR evidence captured (see runbook).

## Next Actions
1. Repository owner runs `tools/enforce_gate_branch_protection.py --check --snapshot docs/evidence/gate-branch-protection/pre.json` with an admin token to confirm the live settings.
2. Follow `docs/runbooks/gate-branch-protection-validation.md` to produce validation PR evidence (including snapshot files) and attach it to issue #2495.
3. Mark the final checkbox above once validation artifacts are archived.
