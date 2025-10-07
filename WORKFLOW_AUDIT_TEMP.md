# Temporary Workflow Audit (Draft)

Date: 2026-10-12

## Naming Compliance Snapshot
- ✅ All active workflows follow the `<area>-<NN>-<slug>.yml` convention with 10-point spacing per family.
- ✅ Each workflow's `name:` field mirrors its filename (title-cased with numeric block preserved).
- ✅ `.github/workflows/archive/` remains absent; legacy self-test wrappers were relocated to `Old/workflows/` for historical reference.

## Final Workflow Set (Issue #2190 + Issue #2379 refresh)
Only the workflows listed below remain visible in the Actions tab. Reusable composites without direct triggers are grouped separately.

### PR Checks
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `pr-10-ci-python.yml` | pull_request, push, workflow_call, workflow_dispatch | Delegates to `reusable-96-ci-lite.yml` for style/type/test coverage plus manual dispatch support.
| `pr-12-docker-smoke.yml` | workflow_call, workflow_dispatch | Deterministic Docker build + smoke test harness.
| `pr-gate.yml` | workflow_call | Aggregates reusable CI/Docker composites into a single gate used by downstream repositories.

### Maintenance & Governance
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `maint-02-repo-health.yml` | schedule, workflow_dispatch | Weekly repository health sweep that records a single run-summary report.
| `maint-30-post-ci-summary.yml` | workflow_run | Posts consolidated CI/Docker run summaries to the workflow step summary.
| `maint-32-autofix.yml` | workflow_run | Follower that applies low-risk autofix commits after CI succeeds.
| `maint-33-check-failure-tracker.yml` | workflow_run | Opens and resolves CI failure tracker issues based on run outcomes.
| `maint-35-repo-health-self-check.yml` | schedule, workflow_dispatch | Governance audit that validates labels/PAT/branch protection and maintains a single failure issue.
| `maint-36-actionlint.yml` | pull_request, push, schedule, workflow_dispatch | Sole workflow lint gate (actionlint via reviewdog).
| `maint-40-ci-signature-guard.yml` | pull_request, push | Verifies signed CI manifests to guard against tampering.
| `maint-41-chatgpt-issue-sync.yml` | workflow_dispatch | Fans out curated topic lists (e.g. `Issues.txt`) into GitHub issues with automatic labeling. |

### Agents
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `agents-consumer.yml` | schedule (hourly), workflow_dispatch | JSON-configurable wrapper that forwards options to `reuse-agents.yml`.
| `agents-43-codex-issue-bridge.yml` | issues, workflow_dispatch | Restored Codex bootstrap automation for label-driven issue handling. |
| `agents-70-orchestrator.yml` | schedule (*/20), workflow_dispatch | Unified agents toolkit entry point (readiness, diagnostics, Codex keepalive). |

### Reusable Composites
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `reuse-agents.yml` | workflow_call | Bridges consumer JSON payloads to the reusable stack.
| `reusable-70-agents.yml` | workflow_call | Reusable agents stack used by `agents-70-orchestrator.yml` and `reuse-agents.yml`.
| `reusable-ci.yml` | workflow_call | General-purpose CI composite (lint, type-check, pytest) for downstream repositories.
| `reusable-99-selftest.yml` | workflow_call | Matrix smoke-test for the reusable CI executor.
| `reusable-90-ci-python.yml` | workflow_call | Primary reusable CI implementation.
| `reusable-92-autofix.yml` | workflow_call | Autofix composite consumed by `maint-32-autofix.yml`.
| `reusable-94-legacy-ci-python.yml` | workflow_call | Legacy CI contract retained for downstream consumers.
| `reusable-96-ci-lite.yml` | workflow_call | Single-job Ruff/mypy/pytest runner used by `pr-10-ci-python.yml` and future gate orchestrators.
| `reusable-97-docker-smoke.yml` | workflow_call | Wrapper that exposes the Docker smoke workflow to orchestration jobs.
| `reusable-docker.yml` | workflow_call | Standalone Docker smoke composite (build + health check) for external consumers.

## Removed in Issue #2190
| Workflow | Status |
|----------|--------|
| `agents-40-consumer.yml`, `agents-41-assign*.yml`, `agents-42-watchdog.yml`, `agents-44-copilot-readiness.yml`, `agents-45-verify-codex-bootstrap-matrix.yml` | Deleted; functionality consolidated into `agents-70-orchestrator.yml` + `reusable-70-agents.yml`.
| `maint-31-autofix-residual-cleanup.yml`, `maint-34-quarantine-ttl.yml`, `maint-37-ci-selftest.yml`, `maint-38-cleanup-codex-bootstrap.yml`, `maint-43-verify-service-bot-pat.yml`, `maint-44-verify-ci-stack.yml`, `maint-45-merge-manager.yml`, `maint-48-selftest-reusable-ci.yml`, `maint-49-stale-prs.yml`, `maint-52-perf-benchmark.yml`, `maint-60-release.yml` | Deleted; maintenance roster trimmed to the Issue #2190 final set.
| `pr-01-gate-orchestrator.yml`, `pr-02-label-agent-prs.yml`, `pr-18-workflow-lint.yml`, `pr-20-selftest-pr-comment.yml`, `pr-30-codeql.yml`, `pr-31-dependency-review.yml`, `pr-path-labeler.yml` | Deleted; PR checks narrowed to the two required pipelines.
| `reuse-agents.yml` (renamed), `agents-consumer.yml` (renamed), `repo-health-self-check.yml` (renamed) | Superseded by the new naming scheme. **2026-10 follow-up:** consumer + bridge reinstated with JSON interface to stay under dispatch limits.

## Archived in Issue #2378

- `maint-90-selftest.yml` relocated to `Old/workflows/` for historical reference when the cron wrapper was retired. `reusable-99-selftest.yml` returned to `.github/workflows/` in Issue #2379 once converted to job-level reuse; the archived copy remains for archaeology.


## Verification
- `pytest tests/test_workflow_*.py` validates naming compliance, inventory coverage, and agent orchestration wiring.
- Manual spot checks confirm `gh workflow list` shows only the Final Workflow Set.

This audit will be deleted once the new documentation in `docs/ci/WORKFLOWS.md` becomes the authoritative catalogue.
