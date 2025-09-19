# CI Workflow Consolidation Plan (Optional Enhancements for #1166)

## Audit Summary (2025-09-19)
The `.github/workflows` directory contains both new reusable workflows and several legacy / overlapping automation files.

### Redundant / Superseded
| Legacy | Reusable Replacement | Action |
| ------ | -------------------- | ------ |
| `autofix.yml` | `reuse-autofix.yml` + `autofix-consumer.yml` | Mark for removal after one release cycle (kept temporarily to avoid breaking external docs / bookmarks). |

### Parallel / Candidate for Future Merge
| Workflow | Notes |
| -------- | ----- |
| `agent-readiness.yml`, `agent-watchdog.yml`, `verify-agent-task.yml` | Functional overlap with `reuse-agents.yml` (watchdog path). Consider collapsing into parameters (e.g., `enable_readiness`) in a later iteration. |
| `codex-bootstrap-diagnostic.yml`, `codex-preflight.yml` | Pre-bootstrap diagnostics; could become a mode in `reuse-agents.yml` (input flag). |
| `verify-codex-bootstrap-matrix.yml` | Specialized matrix verification; keep separate (long-running, matrix heavy). |
| `perf-benchmark.yml` | Performance regression; intentionally standalone (different triggers, resource profile). |

### Keep As-Is
Release, docker, auto-merge enablement, PR status summary, quarantine TTL, failure trackers remain orthogonal to the three reusable workflows.

## Proposed Minimal Consolidation (Current PR Scope)
1. Document redundancy of `autofix.yml` (this file) instead of deleting immediately.
2. Encourage consumers to migrate to `autofix-consumer.yml`.

Rationale: Avoid large diff churn inside the same PR that introduced reusables; allow observability period before deleting legacy file.

## Deletion Timetable (Recommendation)
| File | Earliest Safe Removal | Preconditions |
| ---- | --------------------- | ------------- |
| `autofix.yml` | +2 weeks after merge of PR #1257 | Confirm no external references in docs / badges.

## Future Evolution Ideas
- Parameterise readiness / watchdog / preflight modes inside `reuse-agents.yml` to collapse 3–4 workflows.
- Expose versioned `@v1` tags for remote consumption (convert internal `uses:` paths to fully-qualified refs in downstream repos).
- Add quarantine job implementation tied to `run_quarantine` input in `reuse-ci-python.yml`.

## No Immediate Action Files
All other workflows serve distinct concerns; consolidating now would add complexity without clear maintenance win.

---
Last updated: 2025-09-19
