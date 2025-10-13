# Gate Branch Protection Evidence Archive

Store validation artifacts for issues #2495 and #2527 in this folder. Suggested layout:

```
gate-branch-protection/
  README.md                # Status summary and links to validation evidence
  pre-enforcement.json     # Snapshot captured before running --apply
  enforcement.json         # Snapshot captured during enforcement run
  post-enforcement.json    # Snapshot captured after the Gate check passes
```

Include the JSON snapshots referenced in the runbook alongside any screenshots from
the validation pull request so auditors can confirm the Gate check blocks merges
when failing and allows merges when passing. The scheduled
`health-44-gate-branch-protection` workflow uploads an artifact with the latest
automation snapshots—download it and store the JSON files here for permanence.

## Current archive contents

- `gate-branch-protection/pre-enforcement.json` – Captured 2025-10-13 via the
  public branch endpoint; shows the default branch (`phase-2-dev`) is protected
  and requires **Gate / gate** for non-admin merges.
- `gate-branch-protection/post-enforcement.json` – Mirrors the pre-snapshot
  because no remediation was required.
- `gate-branch-protection/validation-pr-status.json` – Commit status payload for
  PR #2545 confirming the Gate check reports on new pull requests.
- `gate-branch-protection/validation-pr.md` – Narrative describing the
  validation procedure and the associated API queries.

Future audits should refresh these artifacts after any policy update or Gate
workflow rename so the evidence remains current.
