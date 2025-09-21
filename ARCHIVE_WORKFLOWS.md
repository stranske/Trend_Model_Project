# Archived GitHub Workflows (Updated 2025-09-26)

This document records the retirement of legacy agent-related workflows that have now been deleted from `.github/workflows/` in favour of the consolidated reusable pipeline introduced in PR #1257.

## Removed Files
| Legacy Workflow (deleted) | Historical Archive Copy | Replacement Path | Replacement Mode |
|---------------------------|-------------------------|------------------|------------------|
| `.github/workflows/agent-readiness.yml` | (see git history) | `reuse-agents.yml` | `enable_readiness=true` |
| `.github/workflows/agent-watchdog.yml` | (see git history) | `reuse-agents.yml` | `enable_watchdog=true` |
| `.github/workflows/codex-preflight.yml` | (see git history) | `reuse-agents.yml` | `enable_preflight=true` |
| `.github/workflows/codex-bootstrap-diagnostic.yml` | (see git history) | `reuse-agents.yml` | `enable_diagnostic=true` |
| `.github/workflows/verify-agent-task.yml` | (see git history) | `reuse-agents.yml` | `enable_verify_issue=true` |
| `.github/workflows/autofix.yml` | (see git history) | `reuse-autofix.yml` + `autofix-consumer.yml` | `opt_in_label` |
| `.github/workflows/guard-no-reuse-pr-branches.yml` | (in-place archived) | Policy / docs only | n/a |

`archive/*.yml` entries referenced above remain available in git history for forensic review when necessary.

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
| Consider refactor: `autofix-on-failure.yml` → call `reuse-autofix.yml` | TBD | P3 |
| Unify `codex-issue-bridge.yml` with `reuse-agents.yml` bootstrap path | TBD | P3 |

## Verification Checklist
- [x] Archive directory created: `.github/workflows/archive/`
- [x] Stub headers inserted in original workflows marking ARCHIVED status
- [x] Replacements confirmed operational (`reuse-agents.yml` present)

---
Generated as part of workflow hygiene initiative.
