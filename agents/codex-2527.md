<!-- Codex tracking for issue #2527: Enforce Gate branch protection -->

## Status Snapshot
- [ ] Planning brief captured under `docs/planning/issue-2527-gate-branch-protection.md`.
- [ ] Branch protection rule on the default branch lists **Gate / gate** as a required status check with "require branches to be up to date" enabled.
- [ ] Evidence snapshots archived in `docs/evidence/gate-branch-protection/` (pre-check, enforcement, post-check).
- [ ] Validation pull request demonstrates the Gate requirement blocking merge when pending/failing and allowing merge when successful.

## Next Actions
1. Draft or update the planning brief with scope, acceptance criteria, and validation checklist for enforcing Gate on the default branch.
2. Run `tools/enforce_gate_branch_protection.py --check` (optionally with `--apply`) using an admin-scoped token to ensure the rule requires **Gate / gate** and `strict` mode.
3. Capture JSON snapshots (before/after) plus PR evidence per `docs/runbooks/gate-branch-protection-validation.md` and store them under `docs/evidence/gate-branch-protection/`.
4. Update this tracker as milestones are completed and close out issue #2527 once the validation artifacts are in place.
