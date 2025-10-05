# Workflow Naming & Inventory (Issue #2190)

This document tracks the acceptance criteria for Issue #2190. Use it as the authoritative reference for the remaining workflows.

## Naming Policy
- Filenames follow `<area>-<NN>-<slug>.yml` with 10-point spacing per family (`pr-1x`, `maint-0x/3x/4x/9x`, `agents-7x`, `reusable-7x/9x`).
- The workflow `name:` matches the filename rendered in Title Case (`pr-10-ci-python.yml` → `PR 10 CI Python`).
- `tests/test_workflow_naming.py` enforces the policy; rerun `pytest tests/test_workflow_naming.py` after editing workflows.

## Visible Workflows (Actions Tab)
| Filename | `name:` | Purpose |
|----------|---------|---------|
| `.github/workflows/pr-10-ci-python.yml` | `PR 10 CI Python` | Primary CI wrapper (tests, style/type checks, gate aggregation). |
| `.github/workflows/pr-12-docker-smoke.yml` | `PR 12 Docker Smoke` | Deterministic Docker build + smoke tests. |
| `.github/workflows/maint-02-repo-health.yml` | `Maint 02 Repo Health` | Nightly/weekly repository health probe. |
| `.github/workflows/maint-30-post-ci-summary.yml` | `Maint 30 Post CI Summary` | Posts CI/Docker run summaries on `workflow_run`. |
| `.github/workflows/maint-32-autofix.yml` | `Maint 32 Autofix` | Applies autofix commits after CI completes. |
| `.github/workflows/maint-33-check-failure-tracker.yml` | `Maint 33 Check Failure Tracker` | Manages CI failure tracker issues. |
| `.github/workflows/maint-36-actionlint.yml` | `Maint 36 Actionlint` | Sole workflow-lint gate (actionlint via reviewdog). |
| `.github/workflows/maint-40-ci-signature-guard.yml` | `Maint 40 CI Signature Guard` | Validates signed CI manifests. |
| `.github/workflows/maint-90-selftest.yml` | `Maint 90 Selftest` | Manual/weekly caller for `reusable-99-selftest.yml`. |
| `.github/workflows/agents-43-codex-issue-bridge.yml` | `Agents 43 Codex Issue Bridge` | Issue label trigger that bootstraps Codex branches/PRs automatically. |
| `.github/workflows/agents-70-orchestrator.yml` | `Agents 70 Orchestrator` | Unified agents toolkit entry point. |

Only these workflows appear in the Actions UI; everything else is a reusable composite.

## Reusable Composites
| Filename | `name:` | Notes |
|----------|---------|-------|
| `.github/workflows/reusable-90-ci-python.yml` | `Reusable 90 CI Python` | Matrix CI executor used by the PR workflows and self-test matrix. |
| `.github/workflows/reusable-94-legacy-ci-python.yml` | `Reusable 94 Legacy CI Python` | Compatibility shim for downstream consumers. |
| `.github/workflows/reusable-92-autofix.yml` | `Reusable 92 Autofix` | Autofix harness consumed by `maint-32-autofix.yml`. |
| `.github/workflows/reusable-70-agents.yml` | `Reusable 70 Agents` | Readiness, Codex bootstrap, verification, watchdog jobs. |
| `.github/workflows/reusable-99-selftest.yml` | `Reusable 99 Selftest` | Matrix smoke-test of reusable CI features. |

## Trigger Dependencies
- `maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`, and `maint-33-check-failure-tracker.yml` listen for `workflow_run` events from `PR 10 CI Python`, `PR 12 Docker Smoke`, and `Maint 90 Selftest`.
- `Agents 70 Orchestrator` dispatches to `Reusable 70 Agents` and parses extended options via `options_json` to stay under GitHub's 10 input limit.
- `Agents 43 Codex Issue Bridge` acts on `agent:codex` issue labels or manual dispatch to prepare Codex-ready branches and PRs.

## Verification Checklist
- [x] Filenames and `name:` values verified.
- [x] Redundant workflows removed (`agents-4x`, `maint-3x/4x/5x` variants, gate orchestrators, CodeQL, dependency review, labelers).
- [x] Tests updated (`tests/test_workflow_agents_consolidation.py`) to guard the new structure.
- [x] Documentation updated (`docs/agent-automation.md`, `docs/ops/codex-bootstrap-facts.md`, `docs/WORKFLOW_GUIDE.md`, `WORKFLOW_AUDIT_TEMP.md`).

## Quick Validation Commands
- `pytest tests/test_workflow_naming.py` — Ensures filenames and `name:` fields stay aligned with the WFv1 convention.
- `pytest tests/test_automation_workflows.py -k agents` — Spot-checks the agents orchestrator wiring after edits.
- `gh workflow list --limit 20` — Confirm the Actions tab only exposes the workflows listed above.

Run `pytest tests/test_workflow_naming.py` after any workflow change to ensure the guardrails stay green.
