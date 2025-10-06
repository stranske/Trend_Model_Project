# Workflow Topology & Agent Routing Guide (WFv1)

This guide describes the slimmed-down GitHub Actions footprint after Issue #2190. Every workflow now follows the
`<area>-<NN>-<slug>.yml` naming convention with 10-point number gaps so future additions slot in cleanly.

## WFv1 Naming Scheme

| Prefix | Purpose | Active Examples |
| ------ | ------- | ---------------- |
| `pr-` | Pull-request CI wrappers | `pr-10-ci-python.yml`, `pr-12-docker-smoke.yml` |
| `maint-` | Maintenance, governance, and self-tests | `maint-02-repo-health.yml`, `maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`, `maint-33-check-failure-tracker.yml`, `maint-36-actionlint.yml`, `maint-40-ci-signature-guard.yml`, `maint-90-selftest.yml` |
| `agents-` | Agent orchestration entry points | `agents-70-orchestrator.yml` |
| `reusable-` | Reusable composites invoked by other workflows | `reusable-90-ci-python.yml`, `reusable-94-legacy-ci-python.yml`, `reusable-92-autofix.yml`, `reusable-70-agents.yml`, `reusable-99-selftest.yml` |
| `autofix-` assets | Shared configuration for autofix tooling | `autofix-versions.env` |

**Naming checklist**
1. Choose the correct prefix for the workflow's scope.
2. Select a two-digit block that leaves room for future additions (e.g. use another `maint-3x` slot for maintenance jobs).
3. Title-case the workflow name so it matches the filename (`maint-32-autofix.yml` → `Maint 32 Autofix`).
4. Update this guide and `WORKFLOW_AUDIT_TEMP.md` whenever workflows are added, renamed, or removed.

Tests under `tests/test_workflow_naming.py` enforce the naming policy and inventory parity.

## Final Workflow Set

### PR Checks
- **`pr-10-ci-python.yml`** — Single `ci / python` job that runs Black, Ruff, mypy, pytest with coverage, uploads diagnostics, and enforces a coverage minimum while writing a job summary.
- **`pr-12-docker-smoke.yml`** — Docker build and smoke test pipeline. Keeps deterministic caching and publishes summary logs.

### Maintenance & Governance
- **`maint-02-repo-health.yml`** — Nightly/weekly repository health probe with optional workflow_dispatch reruns.
- **`maint-30-post-ci-summary.yml`** — Subscribes to `workflow_run` events from the PR and Docker workflows. Writes a single step-summary block per head SHA.
- **`maint-32-autofix.yml`** — Follower that applies autofix commits or uploads patches after CI completes successfully or fails for lint-only reasons.
- **`maint-33-check-failure-tracker.yml`** — Opens and resolves CI failure tracker issues using signatures derived from the two PR workflows and the self-test caller.
- **`maint-36-actionlint.yml`** — Sole workflow-lint gate. Runs actionlint via reviewdog on PR edits, pushes, weekly cron, and manual dispatch.
- **`maint-40-ci-signature-guard.yml`** — Guards the CI manifest with signed fixture checks.
- **`maint-90-selftest.yml`** — Manual/weekly caller that delegates to `reusable-99-selftest.yml`.

### Agents
- **`agents-70-orchestrator.yml`** — Hourly + manual dispatch entry point for readiness, Codex bootstrap, issue verification, and watchdog sweeps. Delegates to `reusable-70-agents.yml` and accepts extended options via `options_json`.

### Reusable Composites
- **`reusable-90-ci-python.yml`** — Legacy reusable CI matrix retained for self-test coverage while downstream consumers migrate to the single-job workflow.
- **`reusable-94-legacy-ci-python.yml`** — Compatibility shim retained for downstream repositories yet to migrate.
- **`reusable-92-autofix.yml`** — Autofix harness used by `maint-32-autofix.yml`.
- **`reusable-70-agents.yml`** — Reusable agent automation stack.
- **`reusable-99-selftest.yml`** — Matrix self-test covering reusable CI feature flags.

## Trigger Wiring Tips
1. When renaming a workflow, update any `workflow_run` consumers. In this roster that includes `maint-30-post-ci-summary.yml`, `maint-32-autofix.yml`, and `maint-33-check-failure-tracker.yml`.
2. The orchestrator relies on the workflow names, not just filenames. Keep `name:` fields synchronized with filenames to avoid missing triggers.
3. Reusable workflows stay invisible in the Actions tab; top-level consumers should include summary steps for observability.

## Agent Operations
- All historical wrappers were removed. Use **Agents 70 Orchestrator** directly for readiness checks, Codex bootstrap, or watchdog sweeps.
- Optional flags beyond the standard inputs belong in the `options_json` payload; the orchestrator parses it with `fromJson()`.
- The orchestrator maintains PAT priority (`OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`) via the reusable composite.

## Maintenance Playbook
1. PRs rely on the two CI workflows listed above. Keep them green; the post-CI summary will report their status automatically.
2. Monitor `Maint 33 Check Failure Tracker` issues for recurring failures. Closing the issue or re-running the failing workflow clears the tracker.
3. Run `Maint 90 Selftest` manually when tweaking reusable CI inputs to confirm the matrix still passes.
4. Use `Maint 36 Actionlint` workflow_dispatch for ad-hoc validation of complex workflow edits before pushing.

## Additional References
- `.github/workflows/README.md` — Architecture snapshot for the CI + agent stack.
- `docs/ci/WORKFLOWS.md` — Acceptance-criteria checklist for the final workflow set.
- `docs/agent-automation.md` — Detailed description of the agent orchestrator and options.
