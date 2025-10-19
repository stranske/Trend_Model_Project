# Codex bootstrap for Issue #2813

## Acceptance Criteria
- [x] CI coverage artifacts publish with canonical `coverage-<runtime>` names across reusable CI, Gate, and Post-CI.
- [x] Post-CI consumes the Gate coverage payloads and surfaces non-zero metrics plus a delta table in the automated status summary.

## Task List
- [x] Confirm artifact upload/download expectations across reusable CI, Gate, and Post-CI workflows.
- [x] Update Post-CI coverage aggregation to discover nested runtime payloads exported by Gate runs.
- [x] Smoke-test the coverage summarizer locally with nested directories to ensure runtimes are parsed and statistics emitted.

All acceptance criteria satisfied; coverage payloads now survive the Gate hand-off and are summarized in Post-CI.
