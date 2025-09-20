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
- `codex-issue-bridge.yml` – Issue label → bootstrap PR logic (complex orchestrator).
- `reuse-agents.yml` – Consolidated reusable agent pipeline (readiness, preflight, diagnostic, watchdog, bootstrap, verify issue).
- `agent-readiness.yml` (ARCHIVED 2025-09-20) – Superseded by `reuse-agents.yml` (enable_readiness).
- `agent-watchdog.yml` (ARCHIVED 2025-09-20) – Superseded by `reuse-agents.yml` (enable_watchdog).
- `codex-preflight.yml` (ARCHIVED 2025-09-20) – Superseded by `reuse-agents.yml` (enable_preflight).
- `codex-bootstrap-diagnostic.yml` (ARCHIVED 2025-09-20) – Superseded by `reuse-agents.yml` (enable_diagnostic).
- `verify-agent-task.yml` (ARCHIVED 2025-09-20) – Superseded by `reuse-agents.yml` (enable_verify_issue).
- `verify-codex-bootstrap-matrix.yml` – Scenario matrix harness for Codex bootstrap flows (validation / simulation). Potentially classed as debugging but retains agent-specific scope.

### 3. Repository Maintenance & Hygiene
- `stale-prs.yml` – Marks & closes stale PRs.
- `cleanup-codex-bootstrap.yml` – (Not yet read in this draft: needs review; presumed maintenance.)
- `perf-benchmark.yml` – Scheduled performance tracking (benchmark artifact & regression check) – could also fit Category 6.
- `check-failure-tracker.yml` – Opens / closes CI failure issues.
- `pr-status-summary.yml` – Consolidated status comment after CI/Docker runs.

Update: `cleanup-codex-bootstrap.yml` confirmed – prunes stale `agents/codex-issue-*` bootstrap branches older than configurable age (default 14d). Category 3 (Maintenance). Keep; no overlap.

### 4. Automated Debug / Remediation
- `autofix.yml` (DEPRECATED) – Legacy autofix path; replaced by `reuse-autofix.yml` + `autofix-consumer.yml`.
- `reuse-autofix.yml` – Reusable autofix job logic.
- `autofix-consumer.yml` – Consumer invoking `reuse-autofix.yml` on PR events.
- `autofix-on-failure.yml` – Reactive autofix on failing workflows (CI, Docker, Lint, Tests).
- `verify-ci-stack.yml` – Manual trigger to validate CI/Docker/autofix interplay (observational harness).

### 5. Release & Distribution
- `release.yml` – Tag / manual-dispatch PyPI build & publish, changelog generation.

### 6. Performance & Benchmarking
- `perf-benchmark.yml` – (Dual-listed) Scheduled & on-push performance regression detection.

### 7. Reusable Building Blocks
- `reuse-ci-python.yml` – Reusable Python CI executor (matrix + coverage outputs).
- `reuse-autofix.yml` – (Already listed; cross-category usage as building block.)
- `reuse-agents.yml` – (Already listed; central orchestrator.)

### 8. Governance / Policy Enforcement
- `enable-automerge.yml` – Enables auto-merge when risk + from:agent labels present.
- `autoapprove.yml` – Auto-approves small, allowlisted agent PRs.
- `label-agent-prs.yml` – Applies standardized agent labels (pull_request_target hardened).
- `guard-no-reuse-pr-branches.yml` (ARCHIVED 2025-09-20) – Former branch reuse enforcement; policy-only now.
- `pr-path-labeler.yml` – Path‑based labeling for PR taxonomy.
- `dependency-review.yml` – (Also in Category 1; cross-cutting governance.)
- `quarantine-ttl.yml` – (Also in Category 1; test governance.)

## Deprecated / Superseded (Post-Archival Status)
| Workflow | Replacement | Status | Notes |
|----------|-------------|--------|-------|
| agent-readiness.yml | reuse-agents.yml (enable_readiness) | ARCHIVED | Stub & header applied. |
| agent-watchdog.yml | reuse-agents.yml (enable_watchdog) | ARCHIVED | Stub & header applied. |
| codex-preflight.yml | reuse-agents.yml (enable_preflight) | ARCHIVED | Stub & header applied. |
| codex-bootstrap-diagnostic.yml | reuse-agents.yml (enable_diagnostic) | ARCHIVED | Stub & header applied. |
| verify-agent-task.yml | reuse-agents.yml (enable_verify_issue) | ARCHIVED | Stub & header applied. |
| guard-no-reuse-pr-branches.yml | Policy (no automation) | ARCHIVED | In-place archived with no-op job. |
| autofix.yml | reuse-autofix + consumer | Pending removal | Await stabilization window end. |

## Potential Duplications / Overlaps

### CI / Test Execution
- `ci.yml` vs `reusable-ci-python.yml`: `ci.yml` is a thin consumer wrapper. Keep both (consumer + reusable). No consolidation needed.
- `verify-ci-stack.yml` does not run tests; observational. Keep distinct (diagnostic harness) but could merge into a generalized "ops diagnostics" workflow.

### Agent Workflows
- Multiple standalone agent probe/diagnostic workflows replaced by `reuse-agents.yml`. Archive the deprecated ones to reduce noise.
- `codex-issue-bridge.yml` and parts of `reuse-agents.yml` (bootstrap-codex job) share bootstrap themes. Consider future consolidation by adapting `codex-issue-bridge` to invoke `reuse-agents.yml` with appropriate flags (or vice versa). For now keep due to richer branching / fallback logic in bridge.

### Autofix
- Legacy `autofix.yml` vs `reuse-autofix.yml` + `autofix-consumer.yml`. Remove `autofix.yml` once stabilization period (noted in banner) elapses.
- `autofix-on-failure.yml` is complementary (reactive) and should remain; could optionally be refactored to call `reuse-autofix.yml` directly to unify logic.

### Status & Failure Reporting
- `pr-status-summary.yml` and `check-failure-tracker.yml` both respond to workflow_run. Distinct outputs (comment vs issue). Keep both; minimal overlap.

## Consolidation Recommendations
1. Archive deprecated agent workflows (6 listed) after creating an `ARCHIVE/` subfolder or renaming with `.disabled` suffix for historical reference.
2. Schedule removal of `autofix.yml` after the cited 2-week stabilization window (track PR #1257 merge date).
3. Evaluate merging `verify-ci-stack.yml` and adding a diagnostics job into `ci.yml` guarded by a manual `workflow_dispatch` input (optional enhancement).
4. Long-term: unify agent bootstrap path by wrapping `codex-issue-bridge.yml` logic inside a new job in `reuse-agents.yml` to eliminate parallel branching logic.
5. Refactor `autofix-on-failure.yml` to call `reuse-autofix.yml` for single source of truth (pass through head ref context). Not urgent.

## Archival Impact Analysis
- Risk: Minimal; all deprecated workflows point to fully functional replacements already in active use.
- Mitigation: For each archived workflow, add a stub comment with archival date and replacement pointer if kept in repo.

## Next Steps (Proposed Execution Order)
1. (DONE) Create `ARCHIVE_WORKFLOWS.md` summarizing removals.
2. (DONE) Archive deprecated agent workflows & branch reuse guard.
3. Patch `autofix-on-failure.yml` to invoke reusable logic (optional P2).
4. Draft issue: "Unify codex bootstrap flows" capturing consolidation strategy.
5. Schedule removal PR for `autofix.yml` after window expiry.

## Open Questions
- Confirm exact stabilization end date for `autofix.yml` removal.
- Confirm whether `cleanup-codex-bootstrap.yml` has active consumers before classification (not yet inspected in this draft – follow-up read required).

---
This file is a temporary working draft; not intended for long-term retention.
