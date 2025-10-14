# Validation Pull Request — Gate Branch Protection

- **Pull request:** https://github.com/stranske/Trend_Model_Project/pull/2583
- **Validation window:** 2025-10-14 (UTC)

## Evidence collected

1. **Failing Gate run blocks merge**
  - Commit: `3e21785b17b570495e77601ad1422ee1bbfe0927`
  - Workflow run: https://github.com/stranske/Trend_Model_Project/actions/runs/18486687859
  - Status payload: `docs/evidence/gate-branch-protection/validation-pr-status-failing.json`
  - Result: Gate reported `state: failure` (`core-tests-311` leg) and the merge box showed the red "Merging is blocked" banner until the run was retried.
2. **Passing Gate run clears the block**
  - Commit: `6b9cbb9f32a53409a3ddf0e7493e678f5cbd404e`
  - Workflow run: https://github.com/stranske/Trend_Model_Project/actions/runs/18486884063
  - Status payload: `docs/evidence/gate-branch-protection/validation-pr-status-success.json`
  - Result: Gate returned `state: success` and the merge button became available (draft PR kept unmerged).
3. **Autofix lane confirmation**
  - Concurrent `PR 02 Autofix` runs completed successfully on the same branch. No additional required contexts appeared in the commit-status payload, confirming Gate remains the sole blocking check.

## Follow-up actions

- Repeat this validation after any changes to branch protection or the Gate workflow name.
- Update the evidence bundle if future audits require fresh run IDs or UI screenshots.
