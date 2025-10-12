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
| `pr-gate.yml` | `pull_request`, `workflow_dispatch` | Aggregates reusable CI/Docker jobs into the single required gate (`Gate / gate`). |
| `pr-14-docs-only.yml` | `pull_request` (doc paths) | Detects documentation-only diffs and posts a skip notice instead of launching heavier CI. |
| `autofix.yml` | `pull_request` | PR autofix runner delegating to `reusable-92-autofix.yml`; the `apply` job is required. |

### Maintenance & Governance
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `maint-02-repo-health.yml` | Weekly cron, `workflow_dispatch` | Weekly repository health sweep that records a single run-summary report. |
| `maint-post-ci.yml` | `workflow_run` (Gate), `workflow_dispatch` | Consolidated follower that posts Gate summaries, mirrors the autofix sweep, and maintains the rolling `ci-failure` issue. |
| `maint-33-check-failure-tracker.yml` | `workflow_run` (Gate) | Compatibility shell documenting the delegation to `maint-post-ci.yml`. |
| `maint-35-repo-health-self-check.yml` | Weekly cron, `workflow_dispatch` | Governance probe that surfaces label coverage/branch-protection gaps in the step summary. |
| `maint-36-actionlint.yml` | `pull_request`, weekly cron, `workflow_dispatch` | Sole workflow-lint gate (actionlint via reviewdog). |
| `maint-40-ci-signature-guard.yml` | `pull_request`/`push` (`phase-2-dev`) | Verifies the signed Gate manifest. |
| `maint-41-chatgpt-issue-sync.yml` | `workflow_dispatch` | Manual sync turning curated topic lists into labelled issues. |
| `maint-45-cosmetic-repair.yml` | `workflow_dispatch` | Manual pytest + cosmetic fixer that opens a labelled PR when drift is detected. |

### Agents & Automation
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `agents-43-codex-issue-bridge.yml` | `issues`, `workflow_dispatch` | Label-driven helper that prepares Codex bootstrap issues/PRs; does not replace the orchestrator. |
| `agents-44-verify-agent-assignment.yml` | `workflow_call`, `workflow_dispatch` | Issue verification helper reused by the orchestrator and available for ad-hoc checks. |
| `agents-70-orchestrator.yml` | Cron (`*/20 * * * *`), `workflow_dispatch` | Sole automation entry point orchestrating readiness, diagnostics, bootstrap, keepalive, and watchdog jobs. |

### Reusable Composites
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `reuse-agents.yml` | `workflow_call` | Bridges external callers to the reusable agents stack with consistent defaults. |
| `reusable-70-agents.yml` | `workflow_call` | Implements readiness, bootstrap, diagnostics, keepalive, and watchdog jobs. |
| `reusable-ci.yml` | `workflow_call` | General-purpose Python CI composite consumed by Gate and downstream repositories. |
| `reusable-docker.yml` | `workflow_call` | Docker smoke reusable consumed by Gate and external callers. |
| `reusable-92-autofix.yml` | `workflow_call` | Autofix composite shared by `autofix.yml` and `maint-post-ci.yml`. |
| `reusable-99-selftest.yml` | `workflow_call` | Scenario matrix validating the reusable CI executor. |

## Removed in Issue #2466
| Workflow | Status |
|----------|--------|
| `agents-consumer.yml`, legacy `agents-41*`, `agents-42-watchdog.yml` | Deleted; automation now routes exclusively through `agents-70-orchestrator.yml` + `reusable-70-agents.yml`. |
| `maint-31-autofix-residual-cleanup.yml`, `maint-34-quarantine-ttl.yml`, `maint-37-ci-selftest.yml`, `maint-38-cleanup-codex-bootstrap.yml`, `maint-45-merge-manager.yml`, `maint-48-selftest-reusable-ci.yml`, `maint-49-stale-prs.yml`, `maint-52-perf-benchmark.yml`, `maint-60-release.yml` | Archived during the earlier consolidation; list retained here for archaeology. |
| `pr-01-gate-orchestrator.yml`, `pr-02-label-agent-prs.yml`, `pr-18-workflow-lint.yml`, `pr-20-selftest-pr-comment.yml`, `pr-30-codeql.yml`, `pr-31-dependency-review.yml`, `pr-path-labeler.yml` | Deleted; Gate + Autofix now cover PR CI. |

## Verification
- `pytest tests/test_workflow_*.py` validates naming compliance, inventory coverage, and orchestrator wiring.
- Manual spot checks (`gh workflow list`) confirm only the workflows above appear in the Actions UI.

This audit will be deleted once `docs/ci/WORKFLOWS.md` remains the authoritative catalogue for two release cycles.
