# Workflow Topology & Agent Routing Guide (WFv1)

This guide describes the slimmed-down GitHub Actions footprint after Issues #2190 and #2466. Every workflow now follows the
`<area>-<NN>-<slug>.yml` naming convention with 10-point number gaps so future additions slot in cleanly. The Gate workflow
remains the required merge check, and **Agents 70 Orchestrator is the sole automation entry point** for Codex readiness and
bootstrap runs. For the executive summary of buckets, required checks, and automation roles, begin with
[docs/ci/WORKFLOW_SYSTEM.md](ci/WORKFLOW_SYSTEM.md) before diving into the topology details below.

If you need the quick roster of which workflows stay active, which ones retired, and the policy guardrails that bind them,
start with the high-level [Workflow System Overview](ci/WORKFLOW_SYSTEM.md). This guide then dives into naming, routing, and
operational detail for the kept set.

> _Gate rerun trigger:_ this paragraph was touched on 2025-10-13 to force a fresh Gate workflow execution.

## WFv1 Naming Scheme

| Prefix | Purpose | Active Examples |
| ------ | ------- | ---------------- |
| `pr-` | Pull-request CI wrappers | `pr-00-gate.yml`, `pr-02-autofix.yml` |
| `maint-` | Post-CI maintenance and self-tests | `maint-46-post-ci.yml`, `maint-45-cosmetic-repair.yml` |
| `health-` | Repository health & policy checks | `health-40-repo-selfcheck.yml`, `health-41-repo-health.yml`, `health-42-actionlint.yml`, `health-43-ci-signature-guard.yml`, `health-44-gate-branch-protection.yml` |
| `agents-` | Agent orchestration entry points | `agents-70-orchestrator.yml` |
| `reusable-` | Reusable composites invoked by other workflows | `reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`, `reusable-18-autofix.yml`, `reusable-16-agents.yml` |
| `selftest-` | Manual self-tests & experiments | `selftest-reusable-ci.yml` |
| `autofix-` assets | Shared configuration for autofix tooling | `autofix-versions.env` |

**Naming checklist**
1. Choose the correct prefix for the workflow's scope.
2. Select a two-digit block that leaves room for future additions (e.g. use another `maint-3x` slot for maintenance jobs).
3. Title-case the workflow name so it matches the filename (`maint-46-post-ci.yml` → `Maint 46 Post CI`).
4. Update this guide, `docs/ci/WORKFLOWS.md`, and the overview in `docs/ci/WORKFLOW_SYSTEM.md` whenever workflows are added,
   renamed, or removed.

Tests under `tests/test_workflow_naming.py` enforce the naming policy and inventory parity.

## Final Workflow Set

The active roster below mirrors the **Keep** list in the [Workflow System Overview](ci/WORKFLOW_SYSTEM.md). Each entry links back to the filenames under `.github/workflows/` and should be reflected in `docs/ci/WORKFLOWS.md` and the unit tests whenever the inventory changes.

### PR Checks
- **`pr-00-gate.yml`** — Required orchestrator that calls the reusable Python (3.11/3.12) and Docker smoke workflows, then fails fast if any leg does not succeed. A lightweight `detect_doc_only` job mirrors the former PR‑14 filters (Markdown, `docs/`, `assets/`) to skip heavy legs and post the friendly notice when a PR is documentation-only.

_Optional label-gated helper_
- **`pr-02-autofix.yml`** — Opt-in autofix runner (apply only when the `autofix` label is present) delegating to the reusable autofix composite. Leave disabled for standard PRs.

### Maintenance & Repo Health
- **`maint-46-post-ci.yml`** — Follower triggered by the Gate `workflow_run` event that posts consolidated status updates, applies autofix commits or uploads patches, and owns the CI failure-tracker issue/label lifecycle.
- **`health-42-actionlint.yml`** — Sole workflow-lint gate. Runs actionlint via reviewdog on PR edits, pushes, weekly cron, and manual dispatch.
- **`health-43-ci-signature-guard.yml`** — Guards the CI manifest with signed fixture checks.
- **`health-44-gate-branch-protection.yml`** — Enforces branch-protection policy via `tools/enforce_gate_branch_protection.py` when the PAT is configured.

_Additional opt-in utilities_
- **`health-41-repo-health.yml`** — Weekly repository health sweep that writes a single run-summary report, with optional `workflow_dispatch` reruns.
- **`health-40-repo-selfcheck.yml`** — Read-only governance probe that surfaces label coverage and branch-protection visibility gaps in the run summary.
- **`maint-45-cosmetic-repair.yml`** — Manual dispatch utility that runs `pytest -q`, applies guard-gated cosmetic fixes via `scripts/ci_cosmetic_repair.py`, and opens a labelled PR when changes exist.

### Agents & Issues
- **`agents-70-orchestrator.yml`** — 20-minute cron plus manual dispatch entry point for readiness, Codex bootstrap, diagnostics, verification, and keepalive sweeps. Delegates to `reusable-16-agents.yml` and accepts extended options via `options_json`.
- **`agents-63-codex-issue-bridge.yml`** — Label-driven helper that seeds Codex bootstrap PRs and can automatically comment `@codex start`.
- **`agents-63-chatgpt-issue-sync.yml`** — Manual issue fan-out that mirrors curated topic lists into GitHub issues.
- **`agents-64-verify-agent-assignment.yml`** — Workflow-call validator ensuring `agent:codex` issues remain assigned to approved automation accounts.

### Reusable Composites
- **`reusable-10-ci-python.yml`** — Python lint/type/test reusable invoked by Gate and downstream repositories.
- **`reusable-12-ci-docker.yml`** — Docker smoke reusable invoked by Gate and external consumers.
- **`reusable-16-agents.yml`** — Reusable agent automation stack.
- **`reusable-18-autofix.yml`** — Autofix harness used by `maint-46-post-ci.yml` and `pr-02-autofix.yml`.

### Self-tests
- **`selftest-reusable-ci.yml`** — Manual entry point that houses the verification matrix and comment/summary/dual-runtime publication logic.

## Archived & Legacy Workflows

The following workflows were decommissioned during the CI consolidation effort. Keep these references around for historical context only; do not resurrect them without a fresh review. For the authoritative ledger (including verification notes), see [ARCHIVE_WORKFLOWS.md](archive/ARCHIVE_WORKFLOWS.md).

- **`pr-14-docs-only.yml`** — Former docs-only fast path superseded by Gate’s internal detection.
- **`maint-47-check-failure-tracker.yml`** — Replaced by the consolidated post-CI summary in `maint-46-post-ci.yml`.
- **Historical consumer wrappers** — Fully replaced by the orchestrator. Their retirement history now lives in [ARCHIVE_WORKFLOWS.md](archive/ARCHIVE_WORKFLOWS.md).
- **Legacy selftest wrappers** (`selftest-80-pr-comment.yml`, `selftest-82-pr-comment.yml`, `selftest-83-pr-comment.yml`, `selftest-84-reusable-ci.yml`, `selftest-88-reusable-ci.yml`, `selftest-81-reusable-ci.yml`) — Superseded by the consolidated `selftest-reusable-ci.yml`; these wrappers are now removed from `.github/workflows/` and live only in history.

## Trigger Wiring Tips
1. When renaming a workflow, update any `workflow_run` consumers. In this roster that includes `maint-46-post-ci.yml` and `pr-02-autofix.yml`.
2. The orchestrator relies on the workflow names, not just filenames. Keep `name:` fields synchronized with filenames to avoid missing triggers.
3. Reusable workflows stay invisible in the Actions tab; top-level consumers should include summary steps for observability.

### Failure rollup quick reference
- `Maint 46 Post CI` updates the "CI failures in last 24 h" issue labelled `ci-failure`, aggregating failure signatures with links back to the offending Gate runs.
- Auto-heal closes the issue after a full day without repeats while preserving an occurrence history in the body.
- Escalations apply the `priority: high` label once the same signature fires three times.

## Agent Operations
- Use **Agents 70 Orchestrator** for every automation task (readiness checks, Codex bootstrap, diagnostics, keepalive). No other entry points remain; historical consumer shims are documented in [ARCHIVE_WORKFLOWS.md](archive/ARCHIVE_WORKFLOWS.md) for reference only, and the Agent task issue template now auto-labels issues (`agents`, `agent:codex`) so the issue bridge can open the branch/PR before the orchestrator run kicks in.
- Optional flags beyond the standard inputs belong in the `params_json` payload; the orchestrator parses it with `fromJson()` and forwards toggles to `reusable-16-agents.yml`. Include an `options_json` string inside the payload for nested keepalive or cleanup settings when required.
- Provide a PAT when bootstrap needs to push branches. The orchestrator honours PAT priority (`OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`) via the reusable composite.


### Manual dispatch quick steps
1. Open **Actions → Agents 70 Orchestrator → Run workflow**.
2. Supply inputs such as `enable_bootstrap: true` and `bootstrap_issues_label: agent:codex` either via dedicated fields or inside `options_json`.
3. Review the `orchestrate` job summary for readiness tables, bootstrap planner output, verification notes, and links to spawned PRs. Failures provide direct links for triage.
4. For CLI/API usage, reuse the `params_json` example in [docs/ci/WORKFLOWS.md](ci/WORKFLOWS.md#manual-orchestrator-dispatch) and post it directly—either with `gh workflow run agents-70-orchestrator.yml --raw-field params_json="$(cat orchestrator.json)"` or with a REST call such as `curl -X POST ... '{"ref":"phase-2-dev","inputs":{"params_json":"$(cat orchestrator.json)"}}'`. Export `GITHUB_TOKEN` to a PAT or workflow token that can dispatch workflows before invoking the CLI/API call. Mix in individual overrides only when a flag must diverge from the JSON payload.

### Troubleshooting signals
- **Immediate readiness failure** — missing PAT or scope. Inspect the `Authentication` step and rerun with `SERVICE_BOT_PAT`.
- **Bootstrap skipped** — no labelled issues matched `bootstrap_issues_label`. Add the label and rerun.
- **Branch push blocked** — repository protections blocking automation. Grant the PAT required scopes or adjust branch rules.

## Maintenance Playbook
1. PRs rely on the Gate workflow listed above. Keep it green; the post-CI summary will report its status automatically.
2. Monitor failure tracker issues surfaced by `Maint 46 Post CI`; it owns the delegation and auto-heal path end to end.
3. Use `Health 42 Actionlint` (`workflow_dispatch`) for ad-hoc validation of complex workflow edits before pushing.
4. Dispatch `Maint 45 Cosmetic Repair` when you need a curated pytest + hygiene sweep that opens a helper PR with fixes.
5. Run `Maint 47 Disable Legacy Workflows` after archival sweeps to disable any retired workflows that still appear in the Actions UI.

## Additional References
- `.github/workflows/README.md` — Architecture snapshot for the CI + agent stack.
- `docs/ci/WORKFLOWS.md` — Acceptance-criteria checklist for the final workflow set.
- `docs/agent-automation.md` — Detailed description of the agent orchestrator and options.
