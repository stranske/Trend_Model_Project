# Workflow Inventory — Issue #1669 Cleanup

_Date: 2026-10-07_

This inventory captures the authoritative set of GitHub Actions workflows after
the Issue #1669 cleanup. Active files now follow the WFv1 naming scheme and fall
into the `pr-*`, `maint-*`, `agents-*`, or `reusable-*` families. Legacy
placeholders (`guard-no-reuse-pr-branches.yml`, `lint-verification.yml`,
`workflow-lint.yml`) were deleted in this sweep.

## Active Workflow Files

| File | Family | Purpose / Notes |
|------|--------|-----------------|
| `pr-10-ci-python.yml` | PR checks | Compatibility wrapper that preserves the historic **CI** check name while delegating to the reusable matrix runner. |
| `pr-12-docker-smoke.yml` | PR checks | Docker build + smoke tests. |
| `pr-14-dependency-review.yml` | PR checks | Enforces dependency vulnerability review on incoming PRs. |
| `pr-16-path-labeler.yml` | PR checks | Applies taxonomy labels based on touched paths. |
| `pr-18-gate-orchestrator.yml` | PR checks | Orchestrates CI/Docker/actionlint/quarantine TTL for PRs; wraps the reusable consumers. |
| `maint-30-post-ci-summary.yml` | Maintenance | Posts the consolidated CI/Docker status comment. |
| `maint-31-codex-bootstrap-cleanup.yml` | Maintenance | Prunes stale `agents/codex-issue-*` bootstrap branches. |
| `maint-32-autofix.yml` | Maintenance | Consolidated autofix follower reacting to completed CI runs. |
| `maint-33-check-failure-tracker.yml` | Maintenance | Syncs the CI failure tracker issue. |
| `maint-34-quarantine-ttl.yml` | Maintenance | Validates quarantine TTLs and schedules nightly enforcement. |
| `maint-35-repo-health-self-check.yml` | Maintenance | Repository health audit and ops issue updater. |
| `maint-36-actionlint.yml` | Maintenance | Workflow schema lint (reviewdog reporter). |
| `maint-37-codeql.yml` | Maintenance | CodeQL code-scanning analysis (scheduled + PR). |
| `maint-38-stale-prs.yml` | Maintenance | Marks and closes stale pull requests. |
| `maint-39-autofix-residual-cleanup.yml` | Maintenance | Deletes stale autofix scratch branches. |
| `maint-40-ci-signature-guard.yml` | Maintenance | Validates the CI job manifest for signature drift. |
| `maint-41-ci-selftest.yml` | Maintenance | Manual harness for exercising CI failure tracking behaviour. |
| `maint-42-perf-benchmark.yml` | Maintenance | Scheduled performance regression benchmark. |
| `maint-43-selftest-pr-comment.yml` | Maintenance | Scenario matrix for post-CI status comments. |
| `maint-44-selftest-reusable-ci.yml` | Maintenance | Nightly verification for reusable CI feature flags. |
| `maint-45-verify-ci-stack.yml` | Maintenance | Manual diagnostics to verify CI/Docker/autofix interplay. |
| `maint-46-merge-manager.yml` | Maintenance | Unified approval + auto-merge workflow. |
| `maint-50-release.yml` | Maintenance | Manual/tag-triggered release pipeline. |
| `agents-40-consumer.yml` | Agents | Wrapper around reusable agent tooling for readiness/diagnostics drills. |
| `agents-41-assign-and-watch.yml` | Agents | Unified assignment, bootstrap, watchdog, and stale sweep orchestration. |
| `agents-41-assign.yml` | Agents | Thin wrapper forwarding label events into the unified workflow. |
| `agents-42-watchdog.yml` | Agents | Thin wrapper for manual watchdog dispatch. |
| `agents-43-codex-issue-bridge.yml` | Agents | Fallback bridge that provisions Codex bootstrap branches/PRs. |
| `agents-44-chatgpt-issue-sync.yml` | Agents | Imports ChatGPT topic lists and fans out GitHub issues. |
| `agents-45-copilot-readiness.yml` | Agents | Manual probe checking Copilot assignability. |
| `agents-46-label-agent-prs.yml` | Agents | Applies origin/risk labels for automation-authored PRs. |
| `agents-47-verify-codex-bootstrap-matrix.yml` | Agents | Scenario matrix validator for Codex bootstrap flows. |
| `agents-48-verify-service-bot-pat.yml` | Agents | Verifies `SERVICE_BOT_PAT` presence and scopes. |
| `reusable-90-agents.yml` | Reusable | Agent orchestration building block (invoked via wrappers). |
| `reusable-91-autofix.yml` | Reusable | Autofix implementation consumed by the follower workflow or external repos. |
| `reusable-92-ci-python.yml` | Reusable | Matrix Python CI reusable workflow. |
| `reusable-99-selftest.yml` | Reusable | Matrix smoke-test harness for reusable CI features. |
| `reusable-ci-python.yml` | Reusable | Legacy-named reusable CI entry kept for compatibility; points to the WFv1 implementation. |

## Removed / Archived in Issue #1669

| File | Status | Notes |
|------|--------|-------|
| `guard-no-reuse-pr-branches.yml` | Deleted | Archival placeholder retired; policy lives in documentation only. |
| `lint-verification.yml` | Deleted | Legacy stub required check removed after CI style job replaced it. |
| `workflow-lint.yml` | Deleted | Superseded by `maint-36-actionlint.yml`. |

No other files outside the approved families remain under `.github/workflows/`.
