# Auto-merge Smoke Test

This file historically exercised the labeler and the unified merge-manager workflow (auto-approval + auto-merge).
`maint-45-merge-manager.yml` has since been retired in favour of the manual `maint-45-cosmetic-repair.yml` guardrail runner, but
the document remains useful as a smoke test fixture for labeler behaviour.

- Safe path: `docs/**`
- Minimal change to satisfy APPROVE_PATTERNS and size caps.

Runbook:
- Labeler should add `from:codex`, `agent:codex`, `automerge`, and `risk:low` based on branch naming.
- Formerly, the merge manager would auto-approve and auto-merge this PR when thresholds were satisfied. With the workflow
  archived, reviewers should now perform those actions manually while cosmetic repairs run through `maint-45-cosmetic-repair.yml`
  when needed.
