# Workflow Topology & Agent Routing Guide (WFv1)

This guide describes the slimmed-down GitHub Actions footprint after Issues #2190 and #2466. Every workflow now follows the
`<area>-<NN>-<slug>.yml` naming convention with 10-point number gaps so future additions slot in cleanly. The Gate workflow
remains the required merge check, and **Agents 70 Orchestrator is the sole automation entry point** for Codex readiness and
bootstrap runs.

## WFv1 Naming Scheme

| Prefix | Purpose | Active Examples |
| ------ | ------- | ---------------- |
| `pr-` | Pull-request CI wrappers | `pr-00-gate.yml`, `pr-14-docs-only.yml` |
| `maint-` | Post-CI maintenance and self-tests | `maint-30-post-ci.yml`, `maint-33-check-failure-tracker.yml`, `maint-34-cosmetic-repair.yml`, `maint-90-selftest.yml` |
| `health-` | Repository health & policy checks | `health-40-repo-selfcheck.yml`, `health-41-repo-health.yml`, `health-42-actionlint.yml`, `health-43-ci-signature-guard.yml`, `health-44-gate-branch-protection.yml` |
| `agents-` | Agent orchestration entry points | `agents-70-orchestrator.yml` |
| `reusable-` | Reusable composites invoked by other workflows | `reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`, `reusable-92-autofix.yml`, `reusable-70-agents.yml` |
| `autofix-` assets | Shared configuration for autofix tooling | `autofix-versions.env` |

**Naming checklist**
1. Choose the correct prefix for the workflow's scope.
2. Select a two-digit block that leaves room for future additions (e.g. use another `maint-3x` slot for maintenance jobs).
3. Title-case the workflow name so it matches the filename (`maint-30-post-ci.yml` → `Maint Post CI`).
4. Update this guide and `WORKFLOW_AUDIT_TEMP.md` whenever workflows are added, renamed, or removed.

Tests under `tests/test_workflow_naming.py` enforce the naming policy and inventory parity.

## Final Workflow Set

### PR Checks
- **`pr-00-gate.yml`** — Required orchestrator that calls the reusable Python (3.11/3.12) and Docker smoke workflows, then fails fast if any leg does not succeed.
- **`pr-14-docs-only.yml`** — Docs-only detector that posts a skip notice instead of running heavier CI.
- **`autofix.yml`** — PR autofix runner delegating to the reusable autofix composite.

### Maintenance & Governance
- **`health-41-repo-health.yml`** — Weekly repository health sweep that writes a single run-summary report, with optional `workflow_dispatch` reruns.
- **`maint-30-post-ci.yml`** — Follower triggered by the Gate `workflow_run` event that posts consolidated status updates, applies autofix commits or uploads patches, and now owns the CI failure-tracker issue/label lifecycle.
- **`maint-33-check-failure-tracker.yml`** — Lightweight compatibility shell that documents the delegation to `maint-30-post-ci.yml` while legacy listeners migrate.
- **`health-42-actionlint.yml`** — Sole workflow-lint gate. Runs actionlint via reviewdog on PR edits, pushes, weekly cron, and manual dispatch.
- **`health-43-ci-signature-guard.yml`** — Guards the CI manifest with signed fixture checks.
- **`health-40-repo-selfcheck.yml`** — Read-only governance probe that surfaces label coverage and branch-protection visibility gaps in the run summary.
- **`health-44-gate-branch-protection.yml`** — Enforces branch-protection policy via `tools/enforce_gate_branch_protection.py` when the PAT is configured.
- **`maint-34-cosmetic-repair.yml`** — Manual dispatch utility that runs `pytest -q`, applies guard-gated cosmetic fixes via `scripts/ci_cosmetic_repair.py`, and opens a labelled PR when changes exist.

### Agents
- **`agents-70-orchestrator.yml`** — Hourly + manual dispatch entry point for readiness, Codex bootstrap, diagnostics, and keepalive sweeps. Delegates to `reusable-70-agents.yml` and accepts extended options via the `params_json` payload. All prior consumer-style wrappers have been retired.
- **`agents-63-chatgpt-issue-sync.yml`** — Manual issue fan-out that mirrors curated topic lists into GitHub issues.
- **`agents-64-verify-agent-assignment.yml`** — Workflow-call validator ensuring `agent:codex` issues remain assigned to approved automation accounts.

### Reusable Composites
- **`reusable-10-ci-python.yml`** — Python lint/type/test reusable invoked by Gate and any downstream repositories.
- **`reusable-12-ci-docker.yml`** — Docker smoke reusable invoked by Gate and external consumers.
- **`reusable-92-autofix.yml`** — Autofix harness used by `maint-30-post-ci.yml` and `autofix.yml`.
- **`reusable-70-agents.yml`** — Reusable agent automation stack.
- **`selftest-81-reusable-ci.yml`** — Manual self-test covering reusable CI feature flags; dispatch via `workflow_dispatch` when verification is needed.

## Trigger Wiring Tips
1. When renaming a workflow, update any `workflow_run` consumers. In this roster that includes `maint-30-post-ci.yml`, `autofix.yml`, and `maint-33-check-failure-tracker.yml`.
2. The orchestrator relies on the workflow names, not just filenames. Keep `name:` fields synchronized with filenames to avoid missing triggers.
3. Reusable workflows stay invisible in the Actions tab; top-level consumers should include summary steps for observability.

### Failure rollup quick reference
- `Maint Post CI` updates the "CI failures in last 24 h" issue labelled `ci-failure`, aggregating failure signatures with links back to the offending Gate runs.
- Auto-heal closes the issue after a full day without repeats while preserving an occurrence history in the body.
- Escalations apply the `priority: high` label once the same signature fires three times.

## Agent Operations
- Use **Agents 70 Orchestrator** for every automation task (readiness checks, Codex bootstrap, diagnostics, keepalive). No other entry points remain beyond deprecated compatibility shims.
- Optional flags beyond the standard inputs belong in the `params_json` payload; the orchestrator parses it with `fromJson()` and forwards toggles to `reusable-70-agents.yml`. Include an `options_json` string inside the payload for nested keepalive or cleanup settings when required.
- Provide a PAT when bootstrap needs to push branches. The orchestrator honours PAT priority (`OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`) via the reusable composite.

> **Deprecated compatibility wrappers:** `agents-62-consumer.yml` and `agents-consumer.yml` persist only for callers migrating from the historical JSON schema. They remain manual-only and should be phased out in favour of the orchestrator.

### Manual dispatch quick steps
1. Open **Actions → Agents 70 Orchestrator → Run workflow**.
2. Supply inputs such as `enable_bootstrap: true` and `bootstrap_issues_label: agent:codex` either via dedicated fields or inside `params_json`.
3. Review the `orchestrate` job summary for readiness tables, bootstrap planner output, and links to spawned PRs. Failures provide direct links for triage.

### Troubleshooting signals
- **Immediate readiness failure** — missing PAT or scope. Inspect the `Authentication` step and rerun with `SERVICE_BOT_PAT`.
- **Bootstrap skipped** — no labelled issues matched `bootstrap_issues_label`. Add the label and rerun.
- **Branch push blocked** — repository protections blocking automation. Grant the PAT required scopes or adjust branch rules.

## Maintenance Playbook
1. PRs rely on the Gate workflow listed above. Keep it green; the post-CI summary will report its status automatically.
2. Monitor failure tracker issues surfaced by `Maint Post CI`; the legacy `Maint 33` shell simply records the delegation notice.
3. Use `Maint 36 Actionlint` (`workflow_dispatch`) for ad-hoc validation of complex workflow edits before pushing.
4. Dispatch `Maint 45 Cosmetic Repair` when you need a curated pytest + hygiene sweep that opens a helper PR with fixes.

## Additional References
- `.github/workflows/README.md` — Architecture snapshot for the CI + agent stack.
- `docs/ci/WORKFLOWS.md` — Acceptance-criteria checklist for the final workflow set.
- `docs/agent-automation.md` — Detailed description of the agent orchestrator and options.
