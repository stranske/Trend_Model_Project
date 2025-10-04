# Temporary Workflow Audit (Draft)

Date: 2026-10-05

## Categorization Legend
1. Pre-PR / Standard Checks (Quality & Security)
2. Agent Initiation & Support (Codex / Copilot orchestration)
3. Repository Maintenance & Hygiene (scheduled or housekeeping)
4. Automated Debug / Remediation (autofix, failure handlers)
5. Release & Distribution
6. Performance & Benchmarking
7. Reusable Building Blocks (called by other workflows)
8. Governance / Policy Enforcement (labeling, merge policy, stale management)

Additional categories retained so every workflow has a single primary home.

## Inventory by Category

### 1. Pre-PR / Standard Checks
- `pr-01-gate-orchestrator.yml` – Aggregates CI, Docker, actionlint, and quarantine TTL for PR events.
- `pr-10-ci-python.yml` – Core test & coverage gate (consumes `reusable-ci-python.yml`).
- `pr-12-docker-smoke.yml` – Docker image build + smoke tests; complements CI.
- `pr-18-workflow-lint.yml` – Runs actionlint on workflow changes.
- `pr-20-selftest-pr-comment.yml` – Matrix verification for the PR summary bot scenarios.
- `pr-30-codeql.yml` – Static code analysis (push/PR/schedule).
- `pr-31-dependency-review.yml` – Dependency diff vulnerability screening.
- `maint-34-quarantine-ttl.yml` – Validates quarantine expirations (also feeds the gate orchestrator).

### 2. Agent Initiation & Support
- `agents-40-consumer.yml` – Wrapper around the reusable agent toolkit.
- `agents-41-assign-and-watch.yml` – Unified label-driven assignment, Codex bootstrap integration, cross-reference watchdog, and scheduled stale sweep.
- `agents-41-assign.yml` – Forwards issue/PR label events into the unified workflow.
- `agents-42-watchdog.yml` – Preserves the manual watchdog interface while delegating to the unified workflow.
- `agents-43-codex-issue-bridge.yml` – Legacy bootstrap bridge retained for fallback flows.
- `agents-44-copilot-readiness.yml` – Manual readiness probe confirming Copilot can be assigned.
- `agents-45-verify-codex-bootstrap-matrix.yml` – Scenario matrix harness for Codex bootstrap validation.

### 3. Repository Maintenance & Hygiene
- `maint-30-post-ci-summary.yml` – Consolidated status comment after CI/Docker runs.
- `maint-31-autofix-residual-cleanup.yml` – Scheduled cleanup of autofix residual branches and patches.
- `maint-32-autofix.yml` – workflow_run follower that applies small hygiene fixes and trivial failure remediation.
- `maint-33-check-failure-tracker.yml` – Opens / closes CI failure issues.
- `maint-37-ci-selftest.yml` – Intentional fail/succeed pair for guard testing.
- `maint-38-cleanup-codex-bootstrap.yml` – Prunes stale `agents/codex-issue-*` bootstrap branches.
- `maint-41-chatgpt-issue-sync.yml` – Normalizes ChatGPT topic lists into GitHub issues.
- `maint-43-verify-service-bot-pat.yml` – Scheduled guard confirming the automation PAT is still valid.
- `maint-44-verify-ci-stack.yml` – Manual diagnostics harness for CI/Docker/autofix interplay.
- `maint-48-selftest-reusable-ci.yml` – Nightly self-test of the reusable CI workflow matrix.
- `maint-49-stale-prs.yml` – Marks & closes stale PRs.

### 4. Automated Debug / Remediation
- `maint-32-autofix.yml` – Consolidated autofix follower (small fixes + failure remediation).
- `maint-37-ci-selftest.yml` – Guard scenario generator (listed under maintenance but relevant here for diagnostics).
- `maint-44-verify-ci-stack.yml` – Manual verification suite for CI topology (bridge between maintenance + diagnostics).

### 5. Release & Distribution
- `maint-60-release.yml` – Manual/tag-triggered release automation (build, publish, changelog).

### 6. Performance & Benchmarking
- `maint-52-perf-benchmark.yml` – Scheduled/on-push performance regression detection with artifact capture.

### 7. Reusable Building Blocks
- `reusable-ci-python.yml` – Current reusable Python CI executor (matrix + coverage outputs).
- `reusable-legacy-ci-python.yml` – Legacy reusable stack retained for consumers still on WFv0 contract.
- `reusable-autofix.yml` – Reusable autofix job logic.
- `reusable-90-agents.yml` – Reusable agent orchestration stack.
- `reusable-99-selftest.yml` – Matrix smoke-test for reusable CI features.

### 8. Governance / Policy Enforcement
- `maint-40-ci-signature-guard.yml` – Verifies CI job manifests via signed fixtures.
- `maint-45-merge-manager.yml` – Approves and enables auto-merge when allowlist, label, quiet-period, and CI status gates pass.
- `pr-02-label-agent-prs.yml` – Applies standardized agent labels (pull_request_target hardened).
- `pr-path-labeler.yml` – Path-based labeling for taxonomy.
- `maint-34-quarantine-ttl.yml` – (Also in Category 1; enforces quarantine governance.)

## Deprecated / Superseded (Post-Removal Status)
| Workflow | Replacement | Status | Notes |
|----------|-------------|--------|-------|
| agent-readiness.yml | reusable-90-agents.yml (enable_readiness) | REMOVED (2025-09-21) | Historical copy tracked in git history; superseded by unified agent toolkit. |
| agent-watchdog.yml | reusable-90-agents.yml (enable_watchdog) | REMOVED (2025-09-21) | Watchdog functionality merged into `agents-41-assign-and-watch.yml`. |
| codex-preflight.yml | reusable-90-agents.yml (enable_preflight) | REMOVED (2025-09-21) | Diagnostics folded into unified workflow; recover via history if needed. |
| codex-bootstrap-diagnostic.yml | reusable-90-agents.yml (enable_diagnostic) | REMOVED (2025-09-21) | Diagnostic path now exposed through reusable matrix harness. |
| verify-agent-task.yml | reusable-90-agents.yml (enable_verify_issue) | REMOVED (2025-09-21) | Manual verification replaced by watchdog comment + matrix harness. |
| guard-no-reuse-pr-branches.yml | Policy (documentation only) | REMOVED (2026-10-05) | Archived stub deleted; guidance captured in docs. |
| lint-verification.yml | pr-10-ci-python.yml (style job) | REMOVED (2026-10-05) | Legacy stub retired after branch protection migrated to `CI / style`. |
| autoapprove.yml | maint-45-merge-manager.yml | REMOVED (historical) | Replacement recorded in `ARCHIVE_WORKFLOWS.md`. |
| enable-automerge.yml | maint-45-merge-manager.yml | REMOVED (historical) | Replacement recorded in `ARCHIVE_WORKFLOWS.md`. |
| autofix-consumer.yml | maint-32-autofix.yml | REMOVED (2026-02-15) | Small-fix lane lives in consolidated follower. |
| autofix-on-failure.yml | maint-32-autofix.yml | REMOVED (2026-02-15) | Failure remediation merged into consolidated follower. |

## Potential Duplications / Overlaps

### CI / Test Execution
- `pr-10-ci-python.yml` vs `reusable-ci-python.yml`: consumer wrapper vs reusable implementation; both required until all repositories consume WFv1.
- `maint-44-verify-ci-stack.yml` runs diagnostics only; keep separate from CI wrapper.

### Agent Workflows
- Unified flow now covered by the `agents-4x-*` series. Legacy orchestrators live only in history; no files remain under `Old/.github/workflows/`.

### Autofix
- `maint-32-autofix.yml` reacts to CI workflow_run events, orchestrating both hygiene patches and trivial failure remediation via `reusable-autofix.yml`.
- `maint-31-autofix-residual-cleanup.yml` ensures residual branches/artifacts are pruned; keep in tandem.

### Status & Failure Reporting
- `maint-30-post-ci-summary.yml` (PR summary) and `maint-33-check-failure-tracker.yml` (issue tracker) continue to run in parallel with distinct outputs.

## Consolidation Recommendations
1. Monitor adoption of WFv1 reusable stacks and retire `reusable-legacy-ci-python.yml` once all consumers migrate.
2. Evaluate folding `maint-44-verify-ci-stack.yml` into a documented runbook if manual dispatch volume remains low.
3. Periodically audit governance workflows to confirm label/policy names still align with repository conventions.

## Archival Impact Analysis
- Risk: Minimal; all removed workflows have replacements documented in this audit and `ARCHIVE_WORKFLOWS.md`.
- Mitigation: Historical YAML remains in git history; no additional archive directory is required.

## Next Steps (Proposed Execution Order)
1. Update documentation (`docs/WORKFLOW_GUIDE.md`, `.github/workflows/README.md`) to match the renamed slugs.
2. Ensure branch protection references `pr-01-gate-orchestrator.yml` and the consolidated CI jobs.
3. Track downstream consumers of `reusable-legacy-ci-python.yml` and plan migration to WFv1.

## Open Questions
- Confirm whether any external automation references the old `chatgpt-issue-sync.yml` slug; communicate rename as needed.
- Determine retirement timeline for the legacy reusable CI contract once dependent repositories are migrated.

---
This file is a temporary working draft; not intended for long-term retention.
