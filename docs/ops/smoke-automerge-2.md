# Auto-merge Smoke Test 2

This second smoke test file previously validated that labeler → merge-manager (approve + enable) still functioned after recent fixes.
With the merge manager archived, retain this document as a reference scenario for labeler behaviour and manual review steps.

Expectations:
- Branch naming (agents/codex-*) triggers from:codex + agent:codex + automerge + risk:low labels.
- Reviewers now approve this docs-only PR manually.
- Once checks pass, merge via the UI and update the `ci:green` label manually while relying on `maint-46-post-ci.yml` to capture
  the final status summary.
