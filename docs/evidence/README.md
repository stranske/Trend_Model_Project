# Gate Branch Protection Evidence Archive

Store validation artifacts for issue #2495 in this folder. Suggested layout:

```
gate-branch-protection/
  pre-enforcement.json     # Snapshot captured before running --apply
  enforcement.json         # Snapshot captured during enforcement run
  post-enforcement.json    # Snapshot captured after the Gate check passes
```

Include the JSON snapshots referenced in the runbook alongside any screenshots from
the validation pull request so auditors can confirm the Gate check blocks merges
when failing and allows merges when passing.
