# Issue #1667 — Automerge Guardrails

## Source
- **Topic:** Automerge minor changes: confirm rules and labels
- **Trigger:** Documentation-only pull requests should auto-merge when safe
- **Upstream issue:** https://github.com/stranske/Trend_Model_Project/issues/1667

## Goals
1. Keep the "Automerge minor changes" capability for low-risk documentation updates.
2. Require all CI and Docker workflows to finish successfully before the automation unlocks merging.
3. Enforce explicit opt-in via the `automerge` label while rejecting PRs that include any `breaking` labels.
4. Ensure the Merge Manager bot documents why a PR cannot be auto-merged when preconditions are not met.

## Acceptance Criteria
- A documentation-only PR with the `automerge` label merges automatically once CI and Docker report success.
- PRs without the `automerge` label, or with a label containing `breaking`, never auto-merge.
- The automation records clear rationale messages for each guardrail failure condition (missing label, failing status, etc.).
