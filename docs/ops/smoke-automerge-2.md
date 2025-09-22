# Auto-merge Smoke Test 2

This second smoke test file validates that labeler → merge-manager (approve + enable) still function after recent fixes.

Expectations:
- Branch naming (agents/codex-*) triggers from:codex + agent:codex + automerge + risk:low labels.
- Merge manager approves this docs-only PR.
- Merge manager enables squash merging and merges once checks pass, updating the `ci:green` label automatically when gates clear.
