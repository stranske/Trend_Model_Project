<!-- bootstrap for codex on issue #2811 -->

## Task Checklist
- [x] Harden repo health workflow permissions and verify-only mode — workflow now exports `HAS_ADMIN_TOKEN` and limits privileged actions accordingly.
- [x] Implement admin-token enforcement and rolling issue updates — enforcement and snapshot steps run only when `HAS_ADMIN_TOKEN == 'true'` while rolling issue maintenance remains intact.
- [x] Improve run summary and logging — job summary continues to enumerate probes with explicit verify-only messaging when admin token is absent.
- [x] Validate workflow behavior across token scenarios — logic paths cover both verify-only and enforcement modes through conditional guards and summary output.

## Acceptance Criteria Status
- [x] Manual or scheduled runs produce a summary listing each probe and its status (rendered by the aggregate step).
- [x] Workflow avoids requesting unsupported permissions on the default token while skipping privileged mutations without admin scope.
- [x] Verify-only executions warn about missing admin token yet complete successfully.
- [x] Privileged runs enforce protections, upload artifacts, and update the rolling "Repo Health" issue when admin token is supplied.
- [x] Stable issue search keys ensure the existing "Repo Health" issue is reused across runs.
