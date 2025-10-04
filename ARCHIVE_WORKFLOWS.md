# Archived GitHub Workflows (updated 2026-10-07)

This document records the archival and eventual deletion of legacy agent-related workflows now replaced by consolidated reusable pipelines. The most recent sweep (Issue #1419) retired the reusable agent matrix in favour of the focused assigner/watchdog pair. The follow-up sweep for Issue #1669 removed the on-disk archive directory so the history now lives exclusively in git along with this ledger.

## Removed Legacy Files (Cleanup PR for Issue #1259)
All deprecated agent automation workflows were deleted from `.github/workflows/` on 2025-09-21 once the stabilization window for the reusable equivalents closed. Historical copies formerly lived under `.github/workflows/archive/` but that directory was removed on 2026-10-07 as part of the Issue #1669 cleanup. Retrieve any prior YAML from git history when needed.

| Legacy Workflow | Historical Archive Path | Replacement Path | Replacement Mode |
|-----------------|-------------------------|------------------|------------------|
| `agent-readiness.yml` | `archive/agent-readiness.yml` (deleted 2026-10-07) | `reuse-agents.yml` → `agents-41-assign-and-watch.yml` | `enable_readiness=true` |
| `agent-watchdog.yml` | `archive/agent-watchdog.yml` (deleted 2026-10-07) | `reuse-agents.yml` → `agents-41-assign-and-watch.yml` | `enable_watchdog=true` |
| `codex-preflight.yml` | `archive/codex-preflight.yml` (deleted 2026-10-07) | `reuse-agents.yml` (legacy) | `enable_preflight=true` |
| `codex-bootstrap-diagnostic.yml` | `archive/codex-bootstrap-diagnostic.yml` (deleted 2026-10-07) | `reuse-agents.yml` (legacy) | `enable_diagnostic=true` |
| `verify-agent-task.yml` | `archive/verify-agent-task.yml` (deleted 2026-10-07) | `reuse-agents.yml` (legacy) | `enable_verify_issue=true` |

## Additional Archived Workflows
- (2026-02-07) `codex-issue-bridge.yml`, `reuse-agents.yml`, and `agents-consumer.yml` moved to the archive before the assigner/watchdog consolidation. The WFv1 renumbering landed in 2026-09 (`agents-40-consumer.yml`, `agents-41-assign-and-watch.yml`, wrappers, plus `reusable-90-agents.yml`).
- (2026-09-30) Standalone `gate.yml` wrapper deleted (Issue #1657). Aggregation now lives exclusively in the `gate / all-required-green` job inside `pr-10-ci-python.yml`; no archived copy retained because the YAML was invalid.
- (2026-10-05) `autoapprove.yml` and `enable-automerge.yml` permanently retired once `maint-45-merge-manager.yml` proved stable (guard test asserts documentation coverage).
- (2026-10-05) `guard-no-reuse-pr-branches.yml` and `lint-verification.yml` removed after governance documentation and branch protection policies caught up with the consolidated CI stack.
- (2026-10-05) Remaining stub archives under `Old/.github/workflows/` were deleted; historical copies are available via git history and the references below.
- (2026-10-07) `.github/workflows/archive/` removed entirely; Issue #1669 ledger (this file) is now the canonical index for prior workflow names.

## Retired Autofix Wrapper
- Legacy `autofix.yml` (pre-2025) was deleted during the earlier cleanup. As of 2026-02-15 a new consolidated `maint-32-autofix.yml` now drives both small fixes and trivial failure remediation; the former consumer wrappers have been removed.

## Rationale
The 2025 cleanup centralized agent probe, diagnostic, and verification logic into `reuse-agents.yml`. In 2026 this was further simplified: `agents-41-assign-and-watch.yml` now owns assignment, bootstrap, watchdog, and stale sweep duties with thin wrappers preserved for backwards compatibility.

## Rollback Procedure
If a regression is traced to consolidation:
1. Re-enable the specific archived YAML by copying its historical content from git history (pre-archival commit) back into `.github/workflows/`.
2. File an issue documenting the gap vs the reusable job’s behavior.
3. Re-run a targeted `workflow_dispatch` on the restored file for confirmation.

## Follow-Up Tasks
| Task | Owner | Priority |
|------|-------|----------|
| Monitor assigner/watchdog telemetry and add readiness probing only if gap resurfaces | TBD | P3 |

## Verification Checklist
- [x] Archive index maintained: `ARCHIVE_WORKFLOWS.md`
- [x] Stub headers inserted in original workflows marking ARCHIVED status
- [x] Replacements confirmed operational (`agents-41-assign-and-watch.yml` present, wrappers maintained)

---
Generated as part of workflow hygiene initiative.
