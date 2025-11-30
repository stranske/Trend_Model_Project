<!-- bootstrap for codex on issue #2885 -->

## Task list

- [x] Set default-branch protection so the required status checks list exactly **Gate / gate**.
- [x] Extend the Health 41 repo-health workflow to record branch protection, fail when Gate drifts, and surface the summary details.
- [x] Refresh the documentation and runbooks to cover the new branch-protection audit and recovery guidance.

### Acceptance criteria

- [x] Merges are blocked unless Gate is green.
- [x] Health job fails with a clear message if “Gate / gate” is missing or renamed.
- [x] Weekly run summary shows branch protection details.
