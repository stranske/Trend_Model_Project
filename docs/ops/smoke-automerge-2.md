# Auto-merge Smoke Test 2

This second smoke test file is used to validate that labeler → auto-approve → enable-automerge still function after recent fixes.

Expectations:
- Branch naming (agents/codex-*) triggers from:codex + agent:codex + automerge + risk:low labels.
- Auto-approve approves this docs-only PR.
- Enable auto-merge enables squash merging and merges once checks pass.
