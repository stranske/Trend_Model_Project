# Keepalive Status — PR #3813

> **Status:** In progress — documenting the active scope, tasks, and acceptance criteria so keepalive nudges continue until completion.

## Progress updates
- Round 1: Recorded the scope, tasks, and acceptance criteria for the Codex bootstrap work and confirmed the initial placeholder is present.
- Round 2: Reposted the full scope/tasks/acceptance from the PR description to keep keepalive aligned with the outstanding work.

## Scope
- [ ] Surface diagnostics when rank-based selection filters out all candidates so upstream data issues are visible.
- [ ] Preserve existing behaviour for successful selection paths.
- [ ] Ensure diagnostics (warnings or return metadata) are available to callers when selections are empty.

## Tasks
- [ ] Detect empty score sets after filtering and emit a clear warning or error indicating why selection failed.
- [ ] Optionally allow a configurable fallback behaviour (e.g., return metadata explaining the failure) without altering normal successful flows.
- [ ] Add tests that simulate fully filtered inputs and assert the new diagnostics.

## Acceptance criteria
- [ ] Empty selections are accompanied by explicit diagnostics describing the filter condition that led to zero candidates.
- [ ] Default successful paths remain unchanged, and tests cover both empty and non-empty cases.
- [ ] Diagnostics are accessible to callers (e.g., via return metadata or logged warnings).
