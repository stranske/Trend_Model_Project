<!-- bootstrap for codex on issue #2802 -->

## Task Checklist
- [x] Deduplicate keepalive job
- [x] Gate bootstrap by label
- [x] Validate triggers and dry-run behavior

## Acceptance Criteria
- [x] Scheduled run shows a single keepalive job and bootstrap targets labeled issues only
- [x] Run summary lists readiness, preflight, watchdog, and keepalive exactly once
