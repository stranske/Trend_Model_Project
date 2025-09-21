# CI Workflow Consolidation Plan (Optional Enhancements for #1166)

## Audit Summary (2025-09-19)
The `.github/workflows` directory contains both new reusable workflows and several legacy / overlapping automation files.

### Redundant / Superseded (Removed in Cleanup PR for #1259)
| Legacy (now removed) | Reusable Replacement | Removal Rationale |
| -------------------- | -------------------- | ----------------- |
| `autofix-consumer.yml` + `autofix-on-failure.yml` | `autofix.yml` (workflow_run follower) | Small-fix and failure remediation lanes collapsed into a single follower workflow that still delegates to `reuse-autofix.yml`. |
| `agent-readiness.yml` | `reuse-agents.yml` (enable_readiness) | Mode parameter covers readiness path. |
| `agent-watchdog.yml` | `reuse-agents.yml` (watchdog enabled) | Consolidated into single orchestrated workflow. |
| `codex-preflight.yml` | `reuse-agents.yml` (preflight mode) | Folded into parameterized preflight job. |
| `codex-bootstrap-diagnostic.yml` | `reuse-agents.yml` (diagnostic mode) | Unified diagnostics with other agent operations. |
| `verify-agent-task.yml` | `reuse-agents.yml` (verify_issue mode) | Verification now an on-demand mode.

### Parallel / Candidate for Future Merge
| Workflow | Notes |
| -------- | ----- |
| `verify-codex-bootstrap-matrix.yml` | Specialized matrix verification; keep separate (long-running, matrix heavy). |
| `perf-benchmark.yml` | Performance regression; intentionally standalone (different triggers, resource profile). |

### Keep As-Is
Release, docker, auto-merge enablement, PR status summary, quarantine TTL, failure trackers remain orthogonal to the three reusable workflows.

## Consolidation Actions Executed
All previously flagged legacy workflows were deleted in alignment with Issue #1259. As of 2026-02-15 the remaining autofix consumers were collapsed into `autofix.yml`, which still delegates to `reuse-autofix.yml` but now runs exclusively as a CI follower. This concludes the stabilization window referenced in PR #1257.

## Deletion Timetable (Superseded)
Original timetable replaced by immediate removal once validation completed. Retained here for historical context only.

## Future Evolution Ideas
- Parameterise readiness / watchdog / preflight modes inside `reuse-agents.yml` to collapse 3–4 workflows.
- Expose versioned `@v1` tags for remote consumption (convert internal `uses:` paths to fully-qualified refs in downstream repos).
- Add quarantine job implementation tied to `run_quarantine` input in `reuse-ci-python.yml`.

## No Immediate Action Files
All other workflows serve distinct concerns; consolidating now would add complexity without clear maintenance win.

---
Last updated: 2026-02-15
