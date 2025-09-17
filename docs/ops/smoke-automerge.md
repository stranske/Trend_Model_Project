# Auto-merge Smoke Test

This file exists solely to exercise the labeler, auto-approve, and enable auto-merge workflows.

- Safe path: `docs/**`
- Minimal change to satisfy APPROVE_PATTERNS and size caps.

Runbook:
- Labeler should add `from:codex`, `agent:codex`, `automerge`, and `risk:low` based on branch naming.
- Auto-approve should approve this PR if patterns and size are within thresholds.
- Enable auto-merge should enable squash auto-merge and merge after checks pass.
