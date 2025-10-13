# Gate Branch Protection Evidence Archive

Store validation artifacts for issues #2495 and #2527 in this folder. Suggested layout:

```
gate-branch-protection/
  pre-enforcement.json     # Snapshot captured before running --apply
  enforcement.json         # Snapshot captured during enforcement run
  post-enforcement.json    # Snapshot captured after the Gate check passes
```

Include the JSON snapshots referenced in the runbook alongside any screenshots from
the validation pull request so auditors can confirm the Gate check blocks merges
when failing and allows merges when passing. The scheduled
`health-44-gate-branch-protection` workflow uploads an artifact with the latest
automation snapshots—download it and store the JSON files here for permanence.
