# Workflow Topology & Agent Routing Guide (WFv1)

This guide describes the slimmed-down GitHub Actions footprint after Issues #2190 and #2466. Every workflow now follows the
`<area>-<NN>-<slug>.yml` naming convention with 10-point number gaps so future additions slot in cleanly. The Gate workflow
remains the required merge check, while **Agents Issue Intake** is the local bootstrap entry point and the
**Agents Auto-Pilot** route invokes the dispatchable Agents 71 queue selector and Agents 72 worker wrapper for end-to-end issues.
Agents 81 owns active post-Gate delivery; the callable-only Agents 73 definition has no local caller. For the executive
summary of buckets, required checks, and automation roles, begin with
[docs/ci/WORKFLOW_SYSTEM.md](ci/WORKFLOW_SYSTEM.md) before diving into the topology details below.

If you need the quick roster of which workflows stay active, which ones retired, and the policy guardrails that bind them,
start with the high-level [Workflow System Overview](ci/WORKFLOW_SYSTEM.md). This guide then dives into naming, routing, and
operational detail for the kept set.

> _Gate rerun trigger:_ this paragraph was touched on 2025-10-13 to force a fresh Gate workflow execution.

## WFv1 Naming Scheme

| Prefix | Purpose | Active Examples |
| ------ | ------- | ---------------- |
| `pr-` | Pull-request CI wrappers | `pr-00-gate.yml`, `pr-11-ci-smoke.yml` |
| `maint-` | Post-CI maintenance and self-tests | `maint-45-cosmetic-repair.yml`, `maint-46-post-ci.yml`, `maint-47-disable-legacy-workflows.yml`, `maint-50-tool-version-check.yml`, `maint-51-dependency-refresh.yml`, `maint-52-validate-workflows.yml`, `maint-60-release.yml`, `maint-coverage-guard.yml` |
| `health-` | Repository health & policy checks | `health-40-sweep.yml`, `health-40-repo-selfcheck.yml`, `health-41-repo-health.yml`, `health-42-actionlint.yml`, `health-43-ci-signature-guard.yml`, `health-44-gate-branch-protection.yml`, `health-50-security-scan.yml` |
| `agents-` | Agent automation and reusable components | `agents-issue-intake.yml`, `agents-auto-pilot.yml`, `agents-71-codex-belt-dispatcher.yml`, `agents-72-codex-belt-worker-dispatch.yml`, `agents-72-codex-belt-worker.yml`, `agents-73-codex-belt-conveyor.yml`, `agents-80-pr-event-hub.yml`, `agents-81-gate-followups.yml`, `agents-keepalive-sweep.yml`, `agents-guard.yml` |
| `reusable-` | Reusable CI composites invoked by local workflows | `reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`, `reusable-18-autofix.yml` |
| `selftest-` | Manual self-tests & experiments | `selftest-reusable-ci.yml` |
| `autofix.yml` | CI autofix loop | `autofix.yml` |

**Naming checklist**
1. Choose the correct prefix for the workflow's scope.
2. Select a two-digit block that leaves room for future additions (e.g. use another `maint-3x` slot for maintenance jobs).
3. Title-case the workflow name so it matches the filename (`maint-45-cosmetic-repair.yml` → `Maint 45 Cosmetic Repair`).
4. Update this guide, `docs/ci/WORKFLOWS.md`, and the overview in `docs/ci/WORKFLOW_SYSTEM.md` whenever workflows are added,
   renamed, or removed.

Tests under `tests/test_workflow_naming.py` enforce the naming policy and inventory parity.

## Final Workflow Set

The active roster below mirrors the **Keep** list in the [Workflow System Overview](ci/WORKFLOW_SYSTEM.md). Each entry links back to the filenames under `.github/workflows/` and should be reflected in `docs/ci/WORKFLOWS.md` and the unit tests whenever the inventory changes.

### PR Checks
- **`pr-00-gate.yml`** — Required orchestrator that calls the reusable Python (3.11/3.12) and Docker smoke workflows, then fails fast if any leg does not succeed. A lightweight `detect_doc_only` job mirrors the former PR‑14 filters (Markdown, `docs/`, `assets/`) to skip heavy legs and post the friendly notice when a PR is documentation-only.
- **`pr-11-ci-smoke.yml`** — Minimal invariant CI that runs on push/PR to phase-2-dev and main. Installs the project, validates imports, and runs `pytest tests/test_invariants.py` for fast regression detection.

_Inline Gate helper_
- **Gate summary job (`pr-00-gate.yml`)** — Post-CI job that downloads artifacts, computes coverage deltas, runs the label-gated autofix routine, and updates the PR summary comment with a stable marker.

### Maintenance & Repo Health
- **`maint-45-cosmetic-repair.yml`** — Manual dispatch utility that runs `pytest -q`, applies guard-gated cosmetic fixes via `scripts/ci_cosmetic_repair.py`, and opens a labelled PR when changes exist.
- **`maint-46-post-ci.yml`** — Post-CI summary recovery workflow triggered by `workflow_run` on Gate completion. Propagates Gate commit status and posts summaries when the Gate's own summary job doesn't complete.
- **`maint-47-disable-legacy-workflows.yml`** — Manual dispatch utility to disable retired workflows that still appear in the Actions UI.
- **`maint-50-tool-version-check.yml`** — Scheduled + manual dispatch workflow that checks for tool version updates.
- **`maint-51-dependency-refresh.yml`** — Scheduled + manual dispatch workflow for dependency updates.
- **`maint-52-validate-workflows.yml`** — PR/push workflow that validates workflow YAML syntax and structure.
- **`maint-60-release.yml`** — Tag-triggered release workflow for publishing packages.
- **`maint-coverage-guard.yml`** — Daily cron + dispatch workflow that monitors Gate coverage artifacts and maintains the rolling coverage baseline breach issue.
- **`health-40-sweep.yml`** — Weekly sweep that fans out to Actionlint and branch-protection verification. Pull requests trigger the Actionlint leg (paths-filter gated) while schedule/manual runs execute both checks to keep the enforcement snapshots fresh.
- **`health-40-repo-selfcheck.yml`** — Read-only governance probe that surfaces label coverage and branch-protection visibility gaps in the run summary.
- **`health-41-repo-health.yml`** — Weekly repository health sweep that writes a single run-summary report covering stale branches, unassigned issues, and default-branch protection drift, with optional `workflow_dispatch` reruns.
- **`health-42-actionlint.yml`** — Underlying Actionlint job invoked by the sweep (and still runnable via manual dispatch when you need a focused lint dry run).
- **`health-43-ci-signature-guard.yml`** — Guards the CI manifest with signed fixture checks.
- **`health-44-gate-branch-protection.yml`** — Enforces branch-protection policy via `tools/enforce_gate_branch_protection.py` when the PAT is configured (now triggered on PRs or by the consolidated sweep).
- **`health-50-security-scan.yml`** — Security scanning workflow triggered on push, PR, and schedule. Runs vulnerability checks and security audits.

### Agents & Issues
- **`agents-issue-intake.yml`** — Canonical consumer front door that seeds ready-for-review Codex bootstrap PRs on `agent:codex`/`agents:codex` labels, exposes manual dispatch inputs, and calls the shared Workflows implementation.
- **`agents-auto-pilot.yml`** — Label/manual end-to-end controller that dispatches Agents 71, then the Agents 72 dispatch wrapper, and monitors the resulting ready-for-review PR.
- **`agents-71-codex-belt-dispatcher.yml`** — Manual/callable queue selector used by Auto-Pilot to prepare the deterministic `codex/issue-*` branch and mark the issue in progress. It has no cron trigger and does not itself invoke the worker.
- **`agents-72-codex-belt-worker-dispatch.yml`** — Dispatchable wrapper used by Auto-Pilot and operators to invoke the callable worker.
- **`agents-72-codex-belt-worker.yml`** — `workflow_call` component that creates or refreshes the ready-for-review automation PR. It is not directly dispatchable.
- **`agents-73-codex-belt-conveyor.yml`** — `workflow_call` component with no local caller. Do not advertise it as an automatic Gate follower; Agents 81 is the active guarded post-Gate delivery route.
- **`agents-guard.yml`** (aka Health 45 Agents Guard) — PR workflow that validates agent-related labels and permissions.

### Autofix
- **`autofix.yml`** — CI Autofix Loop triggered on `pull_request` and `pull_request_target`. Runs formatting fixes and commits changes back to the PR branch.

### Reusable Composites
- **`reusable-10-ci-python.yml`** — Python lint/type/test reusable invoked by Gate and downstream repositories.
- **`reusable-12-ci-docker.yml`** — Docker smoke reusable invoked by Gate and external consumers.
- **`reusable-18-autofix.yml`** — Autofix harness used by the Gate summary job.

### Self-tests
- **`selftest-reusable-ci.yml`** — Manual entry point that houses the verification matrix and comment/summary/dual-runtime publication logic.

## Archived & Legacy Workflows

The following workflows were decommissioned during the CI consolidation effort. Keep these references around for historical context only; do not resurrect them without a fresh review. For the authoritative ledger (including verification notes), see [ARCHIVE_WORKFLOWS.md](archive/ARCHIVE_WORKFLOWS.md).

- **`pr-14-docs-only.yml`** — Former docs-only fast path superseded by Gate’s internal detection.
- **`maint-47-check-failure-tracker.yml`** — Replaced by the consolidated post-CI summary embedded in the Gate workflow.
- **Historical consumer wrappers** — Fully replaced by shared Workflows orchestration and the local intake/belt entry points. Their retirement history now lives in [ARCHIVE_WORKFLOWS.md](archive/ARCHIVE_WORKFLOWS.md).
- **Legacy selftest wrappers** (`selftest-80-pr-comment.yml`, `selftest-82-pr-comment.yml`, `selftest-83-pr-comment.yml`, `selftest-84-reusable-ci.yml`, `selftest-88-reusable-ci.yml`, `selftest-81-reusable-ci.yml`) — Superseded by the consolidated `selftest-reusable-ci.yml`; these wrappers are now removed from `.github/workflows/` and live only in history.

## Trigger Wiring Tips

1. When renaming a workflow, update any `workflow_run` consumers. In this roster that includes the Gate summary job.
2. Event routers rely on workflow names as well as filenames. Keep `name:` fields synchronized with filenames to avoid missing triggers.
3. Reusable workflows stay invisible in the Actions tab; top-level consumers should include summary steps for observability.

### Failure rollup quick reference
- The Gate summary job updates the "CI failures in last 24 h" issue labelled `ci-failure`, aggregating failure signatures with links back to the offending Gate runs.
- Auto-heal closes the issue after a full day without repeats while preserving an occurrence history in the body.
- Escalations apply the `priority: high` label once the same signature fires three times.

## Agent Operations

- Apply the registry-backed `agent:codex` label or manually dispatch **Agents Issue Intake** in `agent_bridge` mode. The shared implementation remains in `stranske/Workflows`; do not copy it into this consumer repository.
- **Agents Auto-Pilot** invokes Agents 71 and the Agents 72 dispatch wrapper for end-to-end issues. Agents 81 handles guarded post-Gate delivery; Agents 73 is not locally invoked.
- **Agents Keepalive Sweep** periodically re-evaluates stalled non-draft agent PRs through the consolidated gate-followup loop. Labels and lifecycle metadata, not draft state, pause or stage work.
### Manual dispatch quick steps

1. Open **Actions → Agents Issue Intake → Run workflow**.
2. Choose `agent_bridge`, supply the issue number, and leave the bridge agent as `codex` unless routing requires another registered agent.
3. Review the intake summary—or, for `agents:auto-pilot`, the Auto-Pilot, Agents 71, and Agents 72 wrapper summaries—for the spawned ready-for-review PR. Use Agents 81 for the exact-head Gate follow-up state.
4. For CLI usage, run `gh workflow run agents-issue-intake.yml --field mode=agent_bridge --field issue_number=NUMBER` with a token allowed to dispatch workflows.

### Troubleshooting signals

- **Immediate readiness failure** — missing PAT or scope. Inspect the intake run and shared Workflows call, then repair the shared source rather than adding a local shim.
- **Bootstrap skipped** — the issue lacks a registered `agent:*` assignment label. Add the correct label and rerun intake.
- **Branch push blocked** — repository protections blocking automation. Grant the PAT required scopes or adjust branch rules.

## Maintenance Playbook
1. PRs rely on the Gate workflow listed above. Keep it green; the post-CI summary will report its status automatically.
2. Monitor failure tracker issues surfaced by the Gate summary job; it owns the delegation and auto-heal path end to end.
3. Use `Health 40 Sweep` when you want the combined Actionlint + branch-protection sweep, or `Health 42 Actionlint` (`workflow_dispatch`) for an Actionlint-only rehearsal of complex workflow edits before pushing.
4. Dispatch `Maint 45 Cosmetic Repair` when you need a curated pytest + hygiene sweep that opens a helper PR with fixes.
5. Run `Maint 47 Disable Legacy Workflows` after archival sweeps to disable any retired workflows that still appear in the Actions UI.

## Additional References
- `.github/workflows/README.md` — Architecture snapshot for the CI + agent stack.
- `docs/ci/WORKFLOWS.md` — Acceptance-criteria checklist for the final workflow set.
- `docs/agent-automation.md` — Current consumer intake, belt, event-routing, and keepalive topology.
