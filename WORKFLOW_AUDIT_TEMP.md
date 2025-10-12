# Temporary Workflow Audit (Updated for Issue #2466)

Date: 2026-10-12

## Naming Compliance Snapshot
- ✅ All active workflows follow the `<area>-<NN>-<slug>.yml` convention with 10-point spacing per family (exception: `autofix.yml`, reinstated per Issue #2380 as the PR autofix runner).
- ✅ Each workflow's `name:` field mirrors its filename (title-cased with numeric block preserved).
- ✅ `.github/workflows/archive/` remains absent; legacy self-test wrappers live under `Old/workflows/` for archaeology.

## Final Workflow Set
This list mirrors the canonical catalogue in `docs/ci/WORKFLOWS.md` after the Issue #2466 consolidation. Only the workflows below appear in the Actions tab; reusable composites are grouped separately.

### PR Checks
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `pr-00-gate.yml` | pull_request, workflow_dispatch | Aggregates reusable CI/Docker composites into a single required gate for PRs.
| `pr-14-docs-only.yml` | pull_request (doc paths) | Detects documentation-only diffs and posts a skip notice via comment instead of launching heavier CI.
| `autofix.yml` | pull_request | Direct PR autofix runner that delegates to `reusable-92-autofix.yml` and pushes formatting/type hygiene commits when safe.

### Maintenance & Governance
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `maint-02-repo-health.yml` | schedule, workflow_dispatch | Weekly repository health sweep that records a single run-summary report.
| `maint-30-post-ci.yml` | workflow_run | Consolidated post-CI follower that posts Gate summaries, applies low-risk autofix commits, and now owns CI failure-tracker updates.
| `maint-33-check-failure-tracker.yml` | workflow_run | Lightweight compatibility shell delegating all tracker duties to `maint-30-post-ci.yml`.
| `maint-35-repo-health-self-check.yml` | weekly cron, workflow_dispatch | Read-only repo health summary that reports label coverage and branch-protection visibility in the job summary.
| `enforce-gate-branch-protection.yml` | hourly cron, workflow_dispatch | Ensures branch-protection rules stay aligned with the Gate helper (`tools/enforce_gate_branch_protection.py`); exits early when the enforcement PAT is absent.
| `maint-36-actionlint.yml` | pull_request, push, schedule, workflow_dispatch | Sole workflow lint gate (actionlint via reviewdog).
| `maint-40-ci-signature-guard.yml` | pull_request, push | Verifies signed CI manifests to guard against tampering.
| `maint-41-chatgpt-issue-sync.yml` | workflow_dispatch | Fans out curated topic lists (e.g. `Issues.txt`) into GitHub issues with automatic labeling. |
| `maint-45-cosmetic-repair.yml` | workflow_dispatch | Manual pytest run that feeds `scripts/ci_cosmetic_repair.py` to patch guard-gated tolerances/snapshots and open labelled PRs. |

### Agents
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `agents-62-consumer.yml` | workflow_dispatch | Manual-only JSON bridge that calls `reuse-agents.yml`; concurrency guard `agents-62-consumer-${{ github.ref }}` prevents back-to-back dispatch collisions. |
| `agents-43-codex-issue-bridge.yml` | issues, workflow_dispatch | Restored Codex bootstrap automation for label-driven issue handling. |
| `agents-44-verify-agent-assignment.yml` | workflow_call, workflow_dispatch | Validates that `agent:codex` issues remain assigned to an approved agent account before automation runs. |
| `agents-70-orchestrator.yml` | schedule (*/20), workflow_dispatch | Unified agents toolkit entry point (readiness, diagnostics, Codex keepalive). |

### Reusable Composites
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `reuse-agents.yml` | workflow_call | Bridges orchestrator and external callers to the reusable stack.
| `reusable-70-agents.yml` | workflow_call | Reusable agents stack used by `agents-70-orchestrator.yml` and `reuse-agents.yml`.
| `reusable-10-ci-python.yml` | workflow_call | General-purpose CI composite (lint, type-check, pytest) for downstream repositories and Gate.
| `reusable-99-selftest.yml` | workflow_call | Matrix smoke-test for the reusable CI executor.
| `reusable-92-autofix.yml` | workflow_call | Autofix composite consumed by `maint-30-post-ci.yml` and the direct `autofix.yml` PR runner.
| `reusable-94-legacy-ci-python.yml` | workflow_call | Legacy CI contract retained for downstream consumers.
| `reusable-96-ci-lite.yml` | workflow_call | Single-job Ruff/mypy/pytest runner retained for legacy PR 10 experiments and prototype gate research.
| `reusable-97-docker-smoke.yml` | workflow_call | Wrapper that exposes the Docker smoke workflow to orchestration jobs.
| `reusable-12-ci-docker.yml` | workflow_call | Standalone Docker smoke composite (build + health check) for external consumers.

## Removed in Issue #2190
| Workflow | Status |
|----------|--------|
| `agents-40-consumer.yml`, `agents-41-assign*.yml`, `agents-42-watchdog.yml`, `agents-44-copilot-readiness.yml`, `agents-45-verify-codex-bootstrap-matrix.yml` | Deleted; functionality consolidated into `agents-70-orchestrator.yml` + `reusable-70-agents.yml`.
| `maint-31-autofix-residual-cleanup.yml`, `maint-34-quarantine-ttl.yml`, `maint-37-ci-selftest.yml`, `maint-38-cleanup-codex-bootstrap.yml`, `maint-43-verify-service-bot-pat.yml`, `maint-44-verify-ci-stack.yml`, `maint-45-merge-manager.yml`, `selftest-88-reusable-ci.yml`, `maint-49-stale-prs.yml`, `maint-52-perf-benchmark.yml`, `maint-60-release.yml` | Deleted; maintenance roster trimmed to the Issue #2190 final set.
| `pr-01-gate-orchestrator.yml`, `pr-02-label-agent-prs.yml`, `pr-18-workflow-lint.yml`, `selftest-82-pr-comment.yml`, `pr-30-codeql.yml`, `pr-31-dependency-review.yml`, `pr-path-labeler.yml` | Deleted; PR checks narrowed to the two required pipelines.
| `reuse-agents.yml` (renamed), `repo-health-self-check.yml` (renamed) | Superseded by the new naming scheme. **2026-10 follow-up:** orchestrator remains the scheduled dispatch surface; consumer workflow constrained to manual-only usage.

## Archived in Issue #2378

- `maint-90-selftest.yml` relocated to `Old/workflows/` for historical reference when the cron wrapper was retired. `reusable-99-selftest.yml` returned to `.github/workflows/` in Issue #2379 once converted to job-level reuse; the archived copy remains for archaeology.


## Verification
- `pytest tests/test_workflow_*.py` validates naming compliance, inventory coverage, and orchestrator wiring.
- Manual spot checks (`gh workflow list`) confirm only the workflows above appear in the Actions UI.

This audit will be deleted once `docs/ci/WORKFLOWS.md` remains the authoritative catalogue for two release cycles.
