# Archived GitHub Workflows (2025-09-20)

This document records the archival of legacy agent-related workflows now replaced by consolidated reusable pipelines.

## Archived Files
| Original Workflow | Archived Copy | Replacement Path | Replacement Mode |
|-------------------|--------------|------------------|------------------|
| `.github/workflows/agent-readiness.yml` | `archive/agent-readiness.yml` | `reuse-agents.yml` | `enable_readiness=true` |
| `.github/workflows/agent-watchdog.yml` | `archive/agent-watchdog.yml` | `reuse-agents.yml` | `enable_watchdog=true` |
| `.github/workflows/codex-preflight.yml` | `archive/codex-preflight.yml` | `reuse-agents.yml` | `enable_preflight=true` |
| `.github/workflows/codex-bootstrap-diagnostic.yml` | `archive/codex-bootstrap-diagnostic.yml` | `reuse-agents.yml` | `enable_diagnostic=true` |
| `.github/workflows/verify-agent-task.yml` | `archive/verify-agent-task.yml` | `reuse-agents.yml` | `enable_verify_issue=true` |
| `.github/workflows/guard-no-reuse-pr-branches.yml` | (in-place archived) | Policy / docs only | n/a |

## Not Yet Removed
- `autofix.yml` (legacy) – Pending removal after stabilization window (see banner referencing PR #1257). Will be evaluated again after window end date.
- `guard-no-reuse-pr-branches.yml` – Archived (no functional replacement required; governance policy only). Removal candidate after 2025-10-20.

## Rationale
Consolidating agent probe, diagnostic, and verification logic into `reuse-agents.yml` reduces workflow sprawl, centralizes feature flags, and ensures consistent permissions and summary formatting.

## Rollback Procedure
If a regression is traced to consolidation:
1. Re-enable the specific archived YAML by copying its historical content from git history (pre-archival commit) back into `.github/workflows/`.
2. File an issue documenting the gap vs the reusable job’s behavior.
3. Re-run a targeted `workflow_dispatch` on the restored file for confirmation.

## Follow-Up Tasks
| Task | Owner | Priority |
|------|-------|----------|
| Remove `autofix.yml` after stabilization | TBD | P2 |
| Consider refactor: `autofix-on-failure.yml` → call `reuse-autofix.yml` | TBD | P3 |
| Unify `codex-issue-bridge.yml` with `reuse-agents.yml` bootstrap path | TBD | P3 |

## Verification Checklist
- [x] Archive directory created: `.github/workflows/archive/`
- [x] Stub headers inserted in original workflows marking ARCHIVED status
- [x] Replacements confirmed operational (`reuse-agents.yml` present)

---
Generated as part of workflow hygiene initiative.
