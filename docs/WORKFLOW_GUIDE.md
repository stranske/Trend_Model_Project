# Workflow Topology & Agent Routing Guide (WFv1)

This guide describes the slimmed-down GitHub Actions footprint after Issues #2190 and #2466. Every workflow now follows the
`<area>-<NN>-<slug>.yml` naming convention with 10-point number gaps so future additions slot in cleanly. The Gate workflow
remains the required merge check, and **Agents 70 Orchestrator is the sole automation entry point** for Codex readiness and
bootstrap runs.

## WFv1 Naming Scheme

| Prefix | Purpose | Active Examples |
| ------ | ------- | ---------------- |
| `pr-` | Pull-request CI wrappers | `pr-gate.yml`, `pr-14-docs-only.yml` |
| `maint-` | Maintenance, governance, and self-tests | `maint-02-repo-health.yml`, `maint-post-ci.yml`, `maint-33-check-failure-tracker.yml`, `maint-36-actionlint.yml`, `maint-40-ci-signature-guard.yml`, `maint-90-selftest.yml` |
| `agents-` | Agent orchestration entry points | `agents-70-orchestrator.yml` |
| `reusable-` | Reusable composites invoked by other workflows | `reusable-ci.yml`, `reusable-docker.yml`, `reusable-92-autofix.yml`, `reusable-70-agents.yml`, `reusable-99-selftest.yml` |
| `autofix-` assets | Shared configuration for autofix tooling | `autofix-versions.env` |

**Naming checklist**
1. Choose the correct prefix for the workflow's scope.
2. Select a two-digit block that leaves room for future additions (e.g. use another `maint-3x` slot for maintenance jobs).
3. Title-case the workflow name so it matches the filename (`maint-post-ci.yml` → `Maint Post CI`).
4. Update this guide and `WORKFLOW_AUDIT_TEMP.md` whenever workflows are added, renamed, or removed.

Tests under `tests/test_workflow_naming.py` enforce the naming policy and inventory parity.

## Final Workflow Set

### PR Checks
- **`pr-gate.yml`** — Required orchestrator that calls the reusable Python (3.11/3.12) and Docker smoke workflows, then fails fast if any leg does not succeed.
- **`pr-14-docs-only.yml`** — Docs-only detector that posts a skip notice instead of running heavier CI.
- **`autofix.yml`** — PR autofix runner delegating to the reusable autofix composite.

### Maintenance & Governance
- **`maint-02-repo-health.yml`** — Weekly repository health sweep that writes a single run-summary report, with optional `workflow_dispatch` reruns.
- **`maint-post-ci.yml`** — Follower triggered by the Gate `workflow_run` event that posts consolidated status updates, applies autofix commits or uploads patches, and now owns the CI failure-tracker issue/label lifecycle.
- **`maint-33-check-failure-tracker.yml`** — Lightweight compatibility shell that documents the delegation to `maint-post-ci.yml` while legacy listeners migrate.
- **`maint-36-actionlint.yml`** — Sole workflow-lint gate. Runs actionlint via reviewdog on PR edits, pushes, weekly cron, and manual dispatch.
- **`maint-40-ci-signature-guard.yml`** — Guards the CI manifest with signed fixture checks.
- **`maint-90-selftest.yml`** — Manual/weekly caller that delegates to `reusable-99-selftest.yml`.
- **`cosmetic-repair.yml`** — Manual dispatch utility that runs `pytest -q`, applies guard-gated cosmetic fixes via `scripts/ci_cosmetic_repair.py`, and opens a labelled PR when changes exist.

### Agents
- **`agents-70-orchestrator.yml`** — Hourly + manual dispatch entry point for readiness, Codex bootstrap, diagnostics, and keepalive sweeps. Delegates to `reusable-70-agents.yml` and accepts extended options via `options_json`. Legacy wrappers such as `agents-consumer.yml`, `agents-watchdog.yml`, and unconstrained consumer flows have been removed.

### Reusable Composites
- **`reusable-ci.yml`** — Python lint/type/test reusable invoked by Gate and any downstream repositories.
- **`reusable-docker.yml`** — Docker smoke reusable invoked by Gate and external consumers.
- **`reusable-92-autofix.yml`** — Autofix harness used by `maint-32-autofix.yml` and `autofix.yml`.
- **`reusable-70-agents.yml`** — Reusable agent automation stack.
- **`reusable-99-selftest.yml`** — Matrix self-test covering reusable CI feature flags.

## Trigger Wiring Tips
1. When renaming a workflow, update any `workflow_run` consumers. In this roster that includes `maint-post-ci.yml`, `maint-32-autofix.yml`, and `maint-33-check-failure-tracker.yml`.
2. The orchestrator relies on the workflow names, not just filenames. Keep `name:` fields synchronized with filenames to avoid missing triggers.
3. Reusable workflows stay invisible in the Actions tab; top-level consumers should include summary steps for observability.

## Agent Operations
- Use **Agents 70 Orchestrator** for every automation task (readiness checks, Codex bootstrap, diagnostics, keepalive). No other entry points remain.
- Optional flags beyond the standard inputs belong in the `options_json` payload; the orchestrator parses it with `fromJson()` and forwards toggles to `reusable-70-agents.yml`.
- Provide a PAT when bootstrap needs to push branches. The orchestrator honours PAT priority (`OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`) via the reusable composite.

### Manual dispatch quick steps
1. Open **Actions → Agents 70 Orchestrator → Run workflow**.
2. Supply inputs such as `enable_bootstrap: true` and `bootstrap_issues_label: agent:codex` either via dedicated fields or inside `options_json`.
3. Review the `orchestrate` job summary for readiness tables, bootstrap planner output, and links to spawned PRs. Failures provide direct links for triage.

### Troubleshooting signals
- **Immediate readiness failure** — missing PAT or scope. Inspect the `Authentication` step and rerun with `SERVICE_BOT_PAT`.
- **Bootstrap skipped** — no labelled issues matched `bootstrap_issues_label`. Add the label and rerun.
- **Branch push blocked** — repository protections blocking automation. Grant the PAT required scopes or adjust branch rules.

## Maintenance Playbook
1. PRs rely on the Gate workflow listed above. Keep it green; the post-CI summary will report its status automatically.
2. Monitor failure tracker issues surfaced by `Maint Post CI`; the legacy `Maint 33` shell simply records the delegation notice.
3. Run `Maint 90 Selftest` manually when tweaking reusable CI inputs to confirm the matrix still passes.
4. Use `Maint 36 Actionlint` workflow_dispatch for ad-hoc validation of complex workflow edits before pushing.

## Additional References
- `.github/workflows/README.md` — Architecture snapshot for the CI + agent stack.
- `docs/ci/WORKFLOWS.md` — Acceptance-criteria checklist for the final workflow set.
- `docs/agent-automation.md` — Detailed description of the agent orchestrator and options.
