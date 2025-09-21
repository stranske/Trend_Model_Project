# CI Workflow Consolidation Plan (Optional Enhancements for #1166)

## Audit Summary (2025-09-19)
The `.github/workflows` directory contains both new reusable workflows and several legacy / overlapping automation files.

### Redundant / Superseded (Removed in Cleanup PR for #1259)
| Legacy (now removed) | Reusable Replacement | Removal Rationale |
| -------------------- | -------------------- | ----------------- |
| `autofix.yml` | `reuse-autofix.yml` + `autofix-consumer.yml` | Eliminated duplication; stabilization period complete post PR #1257. |
| `agent-readiness.yml` | _(archived)_ | No direct replacement; run ad-hoc GitHub Script checks when needed. |
| `agent-watchdog.yml` | `agent-watchdog.yml` | Standalone watchdog workflow introduced with Issue #1419. |
| `codex-preflight.yml` | _(archived)_ | Manual diagnostics or future targeted scripts as needed. |
| `codex-bootstrap-diagnostic.yml` | _(archived)_ | Superseded by assign-to-agents bootstrap + watchdog pairing. |
| `verify-agent-task.yml` | _(archived)_ | Use issue audit scripts or custom GitHub Script snippets.

### Parallel / Candidate for Future Merge
| Workflow | Notes |
| -------- | ----- |
| `verify-codex-bootstrap-matrix.yml` | Specialized matrix verification; keep separate (long-running, matrix heavy). |
| `perf-benchmark.yml` | Performance regression; intentionally standalone (different triggers, resource profile). |

### Keep As-Is
Release, docker, auto-merge enablement, PR status summary, quarantine TTL, failure trackers remain orthogonal to the three reusable workflows.

## Consolidation Actions Executed
All previously flagged legacy workflows were deleted in alignment with Issue #1259. Consumers now invoke the reusable equivalents (`reuse-autofix.yml`) alongside the new agent pair (`assign-to-agents.yml`, `agent-watchdog.yml`). This concludes the stabilization window referenced in PR #1257.

## Deletion Timetable (Superseded)
Original timetable replaced by immediate removal once validation completed. Retained here for historical context only.

## Future Evolution Ideas
- Monitor whether additional readiness/preflight scripts are required now that `reuse-agents.yml` has been retired in favour of the assigner/watchdog workflows.
- Expose versioned `@v1` tags for remote consumption (convert internal `uses:` paths to fully-qualified refs in downstream repos).
- Add quarantine job implementation tied to `run_quarantine` input in `reuse-ci-python.yml`.

## No Immediate Action Files
All other workflows serve distinct concerns; consolidating now would add complexity without clear maintenance win.

---
Last updated: 2025-09-21
