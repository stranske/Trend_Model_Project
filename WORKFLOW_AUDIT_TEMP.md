# Temporary Workflow Audit (Draft)

Date: 2025-09-20

## Categorization Legend
1. Pre-PR / Standard Checks (Quality & Security)
2. Agent Initiation & Support (Codex / Copilot orchestration)
3. Repository Maintenance & Hygiene (scheduled or housekeeping)
4. Automated Debug / Remediation (autofix, failure handlers)
5. Release & Distribution
6. Performance & Benchmarking
7. Reusable Building Blocks (called by other workflows)
8. Governance / Policy Enforcement (branch reuse, labeling, stale, quarantine)

Additional categories added (5–8) to accommodate all workflows cleanly.

## Inventory by Category

### 1. Pre-PR / Standard Checks
- `ci.yml` – Core test & coverage gate (consumes `reuse-ci-python.yml`).
- `codeql.yml` – Static security analysis.
- `dependency-review.yml` – Dependency diff vulnerability screening.
- `actionlint.yml` – Lints workflow syntax.
- `docker.yml` – Docker image build + test; complements CI.
- `quarantine-ttl.yml` – Ensures test quarantine entries not expired.

### 2. Agent Initiation & Support
- `assign-to-agents.yml` – Label-driven assignment, Codex bootstrap branch/PR creation, trigger comment.
- `agent-watchdog.yml` – Short-horizon diagnostic that confirms a Codex PR cross-reference (or posts a timeout report).
- `verify-codex-bootstrap-matrix.yml` – Scenario matrix harness for Codex bootstrap flows (validation / simulation). Potentially classed as debugging but retains agent-specific scope.
- Legacy orchestrators (`codex-issue-bridge.yml`, `reuse-agents.yml`, plus older probes) now live under `Old/.github/workflows/` for reference after consolidation in Issue #1419.

### 3. Repository Maintenance & Hygiene
- `stale-prs.yml` – Marks & closes stale PRs.
- `cleanup-codex-bootstrap.yml` – (Not yet read in this draft: needs review; presumed maintenance.)
- `perf-benchmark.yml` – Scheduled performance tracking (benchmark artifact & regression check) – could also fit Category 6.
- `check-failure-tracker.yml` – Opens / closes CI failure issues.
- `pr-status-summary.yml` – Consolidated status comment after CI/Docker runs.

Update: `cleanup-codex-bootstrap.yml` confirmed – prunes stale `agents/codex-issue-*` bootstrap branches older than configurable age (default 14d). Category 3 (Maintenance). Keep; no overlap.

### 4. Automated Debug / Remediation
- `reuse-autofix.yml` – Reusable autofix job logic.
- `autofix-consumer.yml` – Consumer invoking `reuse-autofix.yml` on PR events.
- `autofix-on-failure.yml` – Reactive autofix on failing workflows (CI, Docker, Lint, Tests).
- `verify-ci-stack.yml` – Manual trigger to validate CI/Docker/autofix interplay (observational harness).
- *(Removed 2025-09-26)* `autofix.yml` – Legacy standalone workflow retired after stabilization window.

### 5. Release & Distribution
- `release.yml` – Tag / manual-dispatch PyPI build & publish, changelog generation.

### 6. Performance & Benchmarking
- `perf-benchmark.yml` – (Dual-listed) Scheduled & on-push performance regression detection.

### 7. Reusable Building Blocks
- `reuse-ci-python.yml` – Reusable Python CI executor (matrix + coverage outputs).
- `reuse-autofix.yml` – (Already listed; cross-category usage as building block.)

### 8. Governance / Policy Enforcement
- `enable-automerge.yml` – Enables auto-merge when risk + from:agent labels present.
- `autoapprove.yml` – Auto-approves small, allowlisted agent PRs.
- `label-agent-prs.yml` – Applies standardized agent labels (pull_request_target hardened).
- `guard-no-reuse-pr-branches.yml` (ARCHIVED 2025-09-20) – Former branch reuse enforcement; policy-only now.
- `pr-path-labeler.yml` – Path‑based labeling for PR taxonomy.
- `dependency-review.yml` – (Also in Category 1; cross-cutting governance.)
- `quarantine-ttl.yml` – (Also in Category 1; test governance.)

## Deprecated / Superseded (Post-Removal Status)
| Workflow | Replacement | Status | Notes |
|----------|-------------|--------|-------|
| agent-readiness.yml | reuse-agents.yml (enable_readiness) | REMOVED (2025-09-21) | Archived copy retained under `.github/workflows/archive/`; modern flow handled by `assign-to-agents.yml`. |
| agent-watchdog.yml | reuse-agents.yml (enable_watchdog) | REMOVED (2025-09-21) | Watchdog functionality first moved into `reuse-agents.yml`, now superseded by `agent-watchdog.yml`. |
| codex-preflight.yml | reuse-agents.yml (enable_preflight) | REMOVED (2025-09-21) | Archived copy retained for reference; preflight folded into assigner if reintroduced. |
| codex-bootstrap-diagnostic.yml | reuse-agents.yml (enable_diagnostic) | REMOVED (2025-09-21) | Archived copy retained for reference; diagnostics covered by assigner/watchdog pair. |
| verify-agent-task.yml | reuse-agents.yml (enable_verify_issue) | REMOVED (2025-09-21) | Archived copy retained for reference; latest verification runs via watchdog comment. |
| codex-issue-bridge.yml | assign-to-agents.yml | ARCHIVED (2026-02-07) | Moved to `Old/.github/workflows/` after Issue #1419 consolidation. |
| reuse-agents.yml | assign-to-agents.yml + agent-watchdog.yml | ARCHIVED (2026-02-07) | Archived in `Old/.github/workflows/`; parameter matrix deprecated. |
| guard-no-reuse-pr-branches.yml | Policy (no automation) | ARCHIVED | In-place archived with no-op job. |
| autofix.yml | reuse-autofix + consumer | REMOVED (2025-09-21) | Use `autofix-consumer.yml` to call reusable workflow. |

## Potential Duplications / Overlaps

### CI / Test Execution
- `ci.yml` vs `reusable-ci-python.yml`: `ci.yml` is a thin consumer wrapper. Keep both (consumer + reusable). No consolidation needed.
- `verify-ci-stack.yml` does not run tests; observational. Keep distinct (diagnostic harness) but could merge into a generalized "ops diagnostics" workflow.

### Agent Workflows
- Legacy probes previously converged on `reuse-agents.yml`; latest consolidation replaces that matrix with the focused pair `assign-to-agents.yml` + `agent-watchdog.yml`.
- Archived `codex-issue-bridge.yml` and `reuse-agents.yml` remain in `Old/.github/workflows/` for reference but should not receive new consumers.

### Autofix
- Legacy `autofix.yml` has been removed; `autofix-consumer.yml` is the supported entry point calling `reuse-autofix.yml`.
- `autofix-on-failure.yml` is complementary (reactive) and should remain; could optionally be refactored to call `reuse-autofix.yml` directly to unify logic.

### Status & Failure Reporting
- `pr-status-summary.yml` and `check-failure-tracker.yml` both respond to workflow_run. Distinct outputs (comment vs issue). Keep both; minimal overlap.

## Consolidation Recommendations
1. (DONE) Archive deprecated agent workflows (6 listed) with historical copies retained under `.github/workflows/archive/`.
2. (DONE) Remove `autofix.yml` now that the stabilization window following PR #1257 has closed.
3. Evaluate merging `verify-ci-stack.yml` and adding a diagnostics job into `ci.yml` guarded by a manual `workflow_dispatch` input (optional enhancement).
4. Long-term: monitor `assign-to-agents.yml` / `agent-watchdog.yml` telemetry; expand with optional readiness probes only if required (no plan to resurrect `reuse-agents.yml`).
5. Refactor `autofix-on-failure.yml` to call `reuse-autofix.yml` for single source of truth (pass through head ref context). Not urgent.

## Archival Impact Analysis
- Risk: Minimal; all deprecated workflows point to fully functional replacements already in active use.
- Mitigation: For each archived workflow, add a stub comment with archival date and replacement pointer if kept in repo.

## Next Steps (Proposed Execution Order)
1. (DONE) Create `ARCHIVE_WORKFLOWS.md` summarizing removals.
2. (DONE) Archive deprecated agent workflows & branch reuse guard.
3. (DONE) Remove legacy workflow files per consolidation plan.
4. Patch `autofix-on-failure.yml` to invoke reusable logic (optional P2).
5. Draft issue: "Unify codex bootstrap flows" capturing consolidation strategy.

## Open Questions
- (Resolved) Stabilization window for `autofix.yml` concluded with its removal on 2025-09-21.
- Confirm whether `cleanup-codex-bootstrap.yml` has active consumers before classification (not yet inspected in this draft – follow-up read required).

---
This file is a temporary working draft; not intended for long-term retention.
