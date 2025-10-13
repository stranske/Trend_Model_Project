# Gate Branch Protection Evidence Checklist

Use this directory to capture proof that the default branch enforces the **Gate / gate**
status check. Each audit run should populate the following artifacts:

- `pre-enforcement.json` – Output from `tools/enforce_gate_branch_protection.py --check`
  recorded before any changes.
- `enforcement.json` – Snapshot written by the helper when `--apply` is required to
  correct the rule. Omit this file when no update is necessary.
- `post-enforcement.json` – Final verification snapshot after the Gate workflow passes.
- `validation-pr.md` (or screenshot files) – Evidence from the draft validation pull
  request showing Gate blocking merge while failing/pending and allowing merge once the
  run succeeds. Follow the [validation runbook](../../runbooks/gate-branch-protection-validation.md)
  for the capture sequence.

## Current Status (2025-10-13)

- :warning: No snapshots are present yet. Repository administrators must run the helper
  with an admin-scoped token to export the JSON evidence.
- :warning: Validation PR evidence still needs to be collected once Gate enforcement is
  confirmed. See the runbook for the expected screenshots and status text.

Update this file as artifacts are added so auditors can quickly determine whether the
acceptance criteria for issue #2527 have been satisfied.
