# Keepalive Status — PR #3787

> **Status:** Complete — fallback automation path for forked PRs implemented and documented.

## Scope
- [x] Introduce a fallback path that uses GITHUB_TOKEN-only permissions for forked PRs and produces patch artifacts instead of pushing branches when a PAT is missing.
- [x] Preserve the current PAT-powered branch push/comment flow for trusted contexts where secrets are available.
- [x] Ensure autofix status reporting clearly indicates when a fallback mode was used and where maintainers can fetch patches.

## Tasks
- [x] Detect forked contexts or missing ACTIONS_BOT_PAT and switch to a patch-only mode using GITHUB_TOKEN, skipping branch pushes.
- [x] Continue using the PAT path (checkout + push) when secrets are present while keeping step conditions mutual and explicit.
- [x] Emit clear summary/comment text that states whether autofix ran in PAT or fallback mode and links to the produced patch artifact when no push occurs.
- [x] Add coverage in the workflow tests (or mock runs) to confirm both paths succeed without secret access errors on forks.

## Acceptance criteria
- [x] Autofix no longer fails on forked PRs due to missing ACTIONS_BOT_PAT and instead provides a downloadable patch artifact.
- [x] PAT-backed runs still push branches and comments as before, with logs showing which path executed.
- [x] Workflow documentation/comments note the fork-safe fallback behaviour for future maintenance.
