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
| `health-41-repo-health.yml` | schedule, workflow_dispatch | Weekly repository health sweep that records a single run-summary report.
| `maint-30-post-ci.yml` | workflow_run | Consolidated post-CI follower that posts Gate summaries, applies low-risk autofix commits, and now owns CI failure-tracker updates.
| `maint-33-check-failure-tracker.yml` | workflow_run | Lightweight compatibility shell delegating all tracker duties to `maint-30-post-ci.yml`.
| `health-40-repo-selfcheck.yml` | weekly cron, workflow_dispatch | Read-only repo health summary that reports label coverage and branch-protection visibility in the job summary.
| `health-44-gate-branch-protection.yml` | hourly cron, workflow_dispatch | Ensures branch-protection rules stay aligned with the Gate helper (`tools/enforce_gate_branch_protection.py`); exits early when the enforcement PAT is absent.
| `health-42-actionlint.yml` | pull_request, push, schedule, workflow_dispatch | Sole workflow lint gate (actionlint via reviewdog).
| `health-43-ci-signature-guard.yml` | pull_request, push | Verifies signed CI manifests to guard against tampering.
| `agents-63-chatgpt-issue-sync.yml` | workflow_dispatch | Fans out curated topic lists (e.g. `Issues.txt`) into GitHub issues with automatic labeling. |
| `maint-34-cosmetic-repair.yml` | workflow_dispatch | Manual pytest run that feeds `scripts/ci_cosmetic_repair.py` to patch guard-gated tolerances/snapshots and open labelled PRs. |

### Agents
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `agents-62-consumer.yml` | workflow_dispatch | Manual-only JSON bridge that calls `reusable-70-agents.yml`; concurrency guard `agents-62-consumer-${{ github.ref_name }}` prevents back-to-back dispatch collisions. |
| `agents-63-codex-issue-bridge.yml` | issues, workflow_dispatch | Restored Codex bootstrap automation for label-driven issue handling. |
| `agents-64-verify-agent-assignment.yml` | workflow_call, workflow_dispatch | Validates that `agent:codex` issues remain assigned to an approved agent account before automation runs. |
| `agents-70-orchestrator.yml` | schedule (*/20), workflow_dispatch | Unified agents toolkit entry point (readiness, diagnostics, Codex keepalive). |

### Reusable Composites
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `agents-consumer.yml` | `workflow_dispatch` | Manual dispatcher that forwards inputs to `reusable-70-agents.yml` with a lightweight compatibility shim and a concurrency guard (`agents-consumer-${{ github.ref_name }}`). |
| `reusable-70-agents.yml` | `workflow_call` | Single agents composite implementing readiness, bootstrap, diagnostics, keepalive, and watchdog jobs for every caller (orchestrator, consumer, external). Legacy `reuse-agents.yml` was retired during the consolidation. |
| `reusable-10-ci-python.yml` | `workflow_call` | General-purpose Python CI composite consumed by Gate and downstream repositories. |
| `reusable-12-ci-docker.yml` | `workflow_call` | Docker smoke reusable consumed by Gate and external callers. |
| `reusable-92-autofix.yml` | `workflow_call` | Autofix composite shared by `autofix.yml` and `maint-30-post-ci.yml`. |

### Manual Self-Tests
| Workflow | Triggers | Notes |
|----------|----------|-------|
| `selftest-80-pr-comment.yml` | `workflow_dispatch` | PR comment helper that runs `selftest-81-reusable-ci.yml` and updates the specified issue/PR using the `<!-- selftest-80-pr-comment -->` marker. |
| `selftest-81-reusable-ci.yml` | `workflow_dispatch`, `workflow_call` | Core reusable CI self-test matrix exposed for manual runs and downstream wrappers. |
| `selftest-82-pr-comment.yml` | `workflow_dispatch` | Alternate PR comment wrapper with reusable CI messaging tuned for verification runs. |
| `selftest-83-pr-comment.yml` | `workflow_dispatch` | Maintenance-oriented PR comment helper for reusable CI spot checks. |
| `selftest-84-reusable-ci.yml` | `workflow_dispatch` | Summary-focused wrapper that surfaces the verification table in `$GITHUB_STEP_SUMMARY` while delegating to `selftest-81`. |
| `selftest-88-reusable-ci.yml` | `workflow_dispatch` | Dual-runtime summary wrapper feeding multiple Python versions into the reusable matrix before writing a summary. |

## Removed in Issue #2466
| Workflow | Status |
|----------|--------|
| Legacy `agents-41*`, `agents-42-watchdog.yml` | Deleted; automation now routes exclusively through `agents-70-orchestrator.yml` + `reusable-70-agents.yml`. |
| `maint-31-autofix-residual-cleanup.yml`, `maint-34-quarantine-ttl.yml`, `maint-37-ci-selftest.yml`, `maint-38-cleanup-codex-bootstrap.yml`, `maint-45-merge-manager.yml`, `maint-48-selftest-reusable-ci.yml`, `maint-49-stale-prs.yml`, `maint-52-perf-benchmark.yml`, `maint-60-release.yml` | Archived during the earlier consolidation; list retained here for archaeology. |
| `pr-01-gate-orchestrator.yml`, `pr-02-label-agent-prs.yml`, `pr-18-workflow-lint.yml`, `pr-20-selftest-pr-comment.yml`, `pr-30-codeql.yml`, `pr-31-dependency-review.yml`, `pr-path-labeler.yml` | Deleted; Gate + Autofix now cover PR CI. |

## Verification
- `pytest tests/test_workflow_*.py` validates naming compliance, inventory coverage, and orchestrator wiring.
- Manual spot checks (`gh workflow list`) confirm only the workflows above appear in the Actions UI.

This audit will be deleted once `docs/ci/WORKFLOWS.md` remains the authoritative catalogue for two release cycles.
