# Auto-merge Smoke Test

This file exists solely to exercise the labeler and the unified merge-manager workflow (auto-approval + auto-merge).

- Safe path: `docs/**`
- Minimal change to satisfy APPROVE_PATTERNS and size caps.

Runbook:
- Labeler should add `from:codex`, `agent:codex`, `automerge`, and `risk:low` based on branch naming.
- Merge manager should approve this PR if patterns and size are within thresholds.
- Merge manager should enable squash auto-merge and merge after checks pass.
