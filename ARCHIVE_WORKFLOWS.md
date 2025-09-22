# Archived GitHub Workflows (updated 2026-02-07)

This document records the archival and eventual deletion of legacy agent-related workflows now replaced by consolidated reusable pipelines. The most recent sweep (Issue #1419) retired the reusable agent matrix in favour of the focused assigner/watchdog pair.

## Removed Legacy Files (Cleanup PR for Issue #1259)
All deprecated agent automation workflows were deleted from `.github/workflows/` on 2025-09-21 once the stabilization window for the reusable equivalents closed. Historical copies remain under `.github/workflows/archive/` for reference.

| Legacy Workflow | Archived Copy | Replacement Path | Replacement Mode |
|-----------------|---------------|------------------|------------------|
| `agent-readiness.yml` | `archive/agent-readiness.yml` | `reuse-agents.yml` → `assign-to-agents.yml` | `enable_readiness=true` |
| `agent-watchdog.yml` | `archive/agent-watchdog.yml` | `reuse-agents.yml` → `agent-watchdog.yml` | `enable_watchdog=true` |
| `codex-preflight.yml` | `archive/codex-preflight.yml` | `reuse-agents.yml` (legacy) | `enable_preflight=true` |
| `codex-bootstrap-diagnostic.yml` | `archive/codex-bootstrap-diagnostic.yml` | `reuse-agents.yml` (legacy) | `enable_diagnostic=true` |
| `verify-agent-task.yml` | `archive/verify-agent-task.yml` | `reuse-agents.yml` (legacy) | `enable_verify_issue=true` |

## Additional Archived Workflows
- `guard-no-reuse-pr-branches.yml` – Archived in place (no functional replacement required; governance policy only). Removal candidate after 2025-10-20.
- (2026-02-07) `codex-issue-bridge.yml`, `reuse-agents.yml`, and `agents-consumer.yml` moved to `Old/.github/workflows/` after assigner/watchdog consolidation.

## Retired Autofix Wrapper
- Legacy `autofix.yml` (pre-2025) was deleted during the earlier cleanup. As of 2026-02-15 a new consolidated `autofix.yml` now drives both small fixes and trivial failure remediation; the former consumer wrappers have been removed.

## Rationale
The 2025 cleanup centralized agent probe, diagnostic, and verification logic into `reuse-agents.yml`. In 2026 this was further simplified: `assign-to-agents.yml` handles label-driven assignment + Codex bootstrap while `agent-watchdog.yml` supplies the diagnostic signal, reducing optional flags and clarifying ownership.

## Rollback Procedure
If a regression is traced to consolidation:
1. Re-enable the specific archived YAML by copying its historical content from git history (pre-archival commit) back into `.github/workflows/`.
2. File an issue documenting the gap vs the reusable job’s behavior.
3. Re-run a targeted `workflow_dispatch` on the restored file for confirmation.

## Follow-Up Tasks
| Task | Owner | Priority |
|------|-------|----------|
| Consider refactor: `autofix-on-failure.yml` → call `reuse-autofix.yml` | TBD | P3 |
| Monitor assigner/watchdog telemetry and add readiness probing only if gap resurfaces | TBD | P3 |

## Verification Checklist
- [x] Archive directory created: `.github/workflows/archive/`
- [x] Stub headers inserted in original workflows marking ARCHIVED status
- [x] Replacements confirmed operational (`assign-to-agents.yml` + `agent-watchdog.yml` present)

---
Generated as part of workflow hygiene initiative.
