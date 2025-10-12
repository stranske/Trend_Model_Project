# Gate Branch Protection Implementation Plan

## Scope and Key Constraints
- **Target**: Update repository settings so the Gate GitHub Actions workflow (`.github/workflows/pr-gate.yml`) is enforced as a required status check on the default branch.
- **Repository Controls**: All changes to branch protection happen via GitHub UI/API outside this repo; capture procedural steps and validation approach within documentation only.
- **Compatibility**: Ensure no legacy status checks (e.g., "CI") remain required once Gate is enforced to avoid merge blocks on removed jobs.
- **Documentation Update**: CONTRIBUTING guidelines must mention the new requirement without disrupting existing onboarding content.
- **Validation**: Plan for a temporary draft PR to exercise the protection rule without merging extraneous commits to main.

## Acceptance Criteria / Definition of Done
1. Default branch protection rule requires the Gate workflow status, with "Require branches to be up to date" enabled.
2. Obsolete required status checks are removed so only Gate remains enforced.
3. CONTRIBUTING.md includes a note that passing the Gate check is mandatory before merging.
4. A test PR demonstrates that Gate is listed as required and blocks merge while failing, then allows merge after passing.
5. Documentation of the above steps is stored in-repo for future reference.

## Initial Task Checklist
1. **Audit Current Branch Protection**
   - Review existing required status checks and note any legacy entries.
2. **Update Branch Protection Rule**
   - Enable "Require status checks to pass".
   - Select the Gate workflow and enable "Require branches to be up to date".
   - Remove any deprecated contexts such as "CI".
3. **Update Documentation**
   - Add a succinct line to `CONTRIBUTING.md` stating that the Gate check must pass before merging.
4. **Validate via Draft PR**
   - Open a draft PR with a deliberately failing Gate run to verify the merge block.
   - Re-run or fix the failure to confirm the Gate check controls merging.
5. **Record Validation Outcome**
   - Note results of the draft PR test for future audits (e.g., in the issue tracker or repo health checklist).

## Current Status
- ✅ **Documentation updated** – `CONTRIBUTING.md` now explains that the Gate check must pass before merging.
- ✅ **Automation helper** – `tools/enforce_gate_branch_protection.py` provides dry-run, check, and apply modes for owners with the
  appropriate token to configure branch protection without leaving the command line. Unit tests in
  `tests/tools/test_enforce_gate_branch_protection.py` cover the primary scenarios (initial bootstrap, drift detection, and
  clean enforcement).
- ✅ **Scheduled enforcement** – `.github/workflows/health-44-gate-branch-protection.yml` runs nightly and on demand. It applies the
  enforcement automatically whenever the optional `BRANCH_PROTECTION_TOKEN` secret is present, otherwise it exits early with a
  clear reminder to configure the token.
- 🔄 **Validation draft PR** – Pending repository-owner action. Follow the runbook in
  `docs/runbooks/gate-branch-protection-validation.md` to capture evidence of the Gate check blocking merges when failing and
  allowing merges when passing.
