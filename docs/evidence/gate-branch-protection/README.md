# Gate Branch Protection Evidence Bundle

Use this directory to store JSON snapshots, screenshots, and supporting notes that
prove the Gate status check is enforced as a required context on the default branch.
All artifacts should map directly to the validation runbook so auditors can replay
the configuration history.

## Required Artifacts
- `pre-enforcement.json` – output from `tools/enforce_gate_branch_protection.py --snapshot`
  (or `gh api`) showing the state before enforcement.
- `enforcement.json` – snapshot captured immediately after running the helper with
  `--apply`, if changes were necessary.
- `verification.json` – snapshot from the post-enforcement `--check` run (this is
  also emitted by the scheduled `health-44` workflow).
- `validation-pr.md` – short log linking to the draft PR that demonstrated Gate
  blocking merges while failing and unblocking once green. Include timestamps and
  direct URLs to the workflow runs.
- `screenshots/` – optional folder containing images of the GitHub UI while the
  Gate check was red (merge blocked) and green (merge permitted).

## Maintenance Tips
- Keep only the latest confirmed evidence; archive older bundles under a dated
  subdirectory if historical records must be preserved.
- Redact or omit any sensitive information (access tokens, branch names for
  private work) before committing evidence files.
- After updating this bundle, cross-reference the records in
  `docs/runbooks/gate-branch-protection-validation.md` to ensure the runbook reflects
  the latest validation steps.
