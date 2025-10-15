# Gate Enforcement Plan for `phase-2-dev`

## Scope and Key Constraints
- Apply branch protection on the default branch (`phase-2-dev`) so that the **Gate / gate** status check is mandatory before merge.
- Honor repository automation conventions: Health-44 (`.github/workflows/health-44-gate-branch-protection.yml`) performs verification and optional enforcement when a `BRANCH_PROTECTION_TOKEN` secret with admin scope is configured.
- Maintain compatibility with existing CI behaviour: docs-only PRs still short-circuit heavy jobs while reporting Gate success, and all other workflows remain optional.
- Ensure remediation steps respect access boundaries—manual enforcement must use admin-scoped tokens, while monitoring workflows run with default permissions unless elevated via secrets.

## Acceptance Criteria / Definition of Done
1. **Health-44 Verification**
   - A scheduled or manual run of Health-44 completes successfully on `phase-2-dev`, reporting that Gate is the only required status check.
2. **Branch Protection Configuration**
   - Repository settings show Gate as a required check for the default branch with “Require branches to be up to date before merging” enabled.
3. **Test Pull Request Confirmation**
   - A temporary PR targeting `phase-2-dev` lists **Gate / gate** under “Required checks” and blocks merge until it succeeds.
4. **Documentation Update**
   - The workflow system overview documents the enforcement and recovery procedure so future maintainers can replicate the setup.

## Initial Task Checklist
- [ ] Confirm whether an admin PAT (`BRANCH_PROTECTION_TOKEN`) is available; if not, plan for manual enforcement while leaving Health-44 in observe-only mode.
- [ ] Trigger Health-44 (`gh workflow run "Health 44 Gate Branch Protection" --ref phase-2-dev`) to capture the current status and artifacts.
- [ ] Review branch protection settings via the GitHub UI or `python tools/enforce_gate_branch_protection.py --repo <owner>/<repo> --branch phase-2-dev --check`.
- [ ] If Gate is not required, enforce it using the script (`--apply`) or by configuring the PAT for Health-44 and re-running the workflow.
- [ ] Re-run Health-44 to validate enforcement and archive the verification snapshot.
- [ ] Open a throwaway PR against `phase-2-dev` to confirm Gate appears as a required check; close it after verification.
- [ ] Record remediation steps and any tokens or permissions used in the workflow overview documentation.
