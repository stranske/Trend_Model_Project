# Validation Pull Request — Gate Branch Protection

- **Pull request:** https://github.com/stranske/Trend_Model_Project/pull/2665
- **Validation window:** 2025-10-16 (UTC)

## Evidence collected

1. **Branch protection now requires both contexts**
   - Snapshot: `docs/evidence/gate-branch-protection/verification-run-18548574371.json`
   - Result: Health 44 captured the post-enforcement state with both `Gate / gate` **and** `Health 45 Agents Guard / Enforce agents workflow protections` marked required and `strict` enabled on `phase-2-dev`.
2. **Enforcement run applied missing context**
   - Workflow run: https://github.com/stranske/Trend_Model_Project/actions/runs/18548574371
   - Snapshot: `docs/evidence/gate-branch-protection/enforcement-run-18548574371.json`
   - Result: Health 44 ran with `--apply`, added the agents guard context, and confirmed `strict` was already true.
3. **Verification fails when the guard is absent**
   - Workflow run: https://github.com/stranske/Trend_Model_Project/actions/runs/18548480690
   - Snapshot: `docs/evidence/gate-branch-protection/verification-run-18548480690.json`
   - Result: Health 44 (check mode) failed after reporting drift: only Gate was required and the `strict` flag could not be confirmed.

## Follow-up actions

- Refresh this validation bundle if the default branch, workflow names, or required check set changes again.
- Capture a new commit-status payload once PR traffic exercises the dual required checks so the status API evidence matches the enforcement snapshots.
