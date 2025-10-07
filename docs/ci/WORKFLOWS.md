# Workflow Catalog & Contributor Quick Start

Use this page as the canonical reference for CI workflow naming, inventory, and
local guardrails. It consolidates the requirements from Issues #2190 and #2202
and adds guidance for contributors who need to stage new workflows or dispatch
the agents toolkit.

## Naming Policy & Number Ranges

- Store workflows under `.github/workflows/` and follow the
  `<area>-<NN>-<slug>.yml` convention.
- Reserve number bands per area so future additions remain grouped:

  | Prefix | Number slots | Usage | Notes |
  |--------|--------------|-------|-------|
  | `pr-` | `10–19` | Pull-request gates | `pr-10-ci-python.yml` is the primary style/test gate; keep space for specialised PR jobs (Docker, docs).
  | `maint-` | `00–49` and `90s` | Scheduled/background maintenance | Low numbers for repo hygiene, 30s/40s for post-CI and guards, 90 for self-tests calling reusable matrices.
  | `agents-` | `40s` & `70s` | Agent bootstrap/orchestration | `43` bridges issues to Codex; `70` is the manual/scheduled orchestrator.
  | `reusable-` | `70s` & `90s` | Composite workflows invoked by others | Keep 90s for CI executors, 70s for agent composites.

- Match the `name:` field to the filename rendered in Title Case
  (`pr-10-ci-python.yml` → `PR 10 CI Python`).
- `tests/test_workflow_naming.py` enforces this policy—rerun it after modifying
  or adding workflows.
- When introducing a new workflow choose the lowest unused slot inside the
  appropriate band, update this document, and add the workflow to the relevant
  section below.

## Workflow Inventory

### Required PR gates (block merges by default)

| Workflow | Trigger(s) | Why it matters |
|----------|------------|----------------|
| `pr-10-ci-python.yml` (`PR 10 CI Python`) | `pull_request`, `workflow_dispatch` | Unified style/type/test gate (Black, Ruff, mypy, pytest + coverage upload).
| `pr-12-docker-smoke.yml` (`PR 12 Docker Smoke`) | `pull_request`, `workflow_dispatch` | Deterministic Docker build followed by image smoke tests.

These jobs must stay green for PRs to merge. The post-CI maintenance jobs below
listen to their `workflow_run` events.

### Maintenance & observability (scheduled/optional reruns)

| Workflow | Trigger(s) | Purpose |
|----------|------------|---------|
| `maint-02-repo-health.yml` (`Maint 02 Repo Health`) | Weekly cron, manual | Reports stale branches & unassigned issues.
| `maint-30-post-ci-summary.yml` (`Maint 30 Post CI Summary`) | `workflow_run` (PR 10/12) | Publishes consolidated CI status for active PRs.
| `maint-32-autofix.yml` (`Maint 32 Autofix`) | `workflow_run` (PR 10/12 & Maint 90) | Applies formatter/type-hygiene autofixes after CI completes.
| `maint-33-check-failure-tracker.yml` (`Maint 33 Check Failure Tracker`) | `workflow_run` (PR 10/12 & Maint 90) | Manages CI failure-tracker issues.
| `maint-36-actionlint.yml` (`Maint 36 Actionlint`) | `pull_request`, weekly cron, manual | Sole workflow-lint gate (actionlint via reviewdog).
| `maint-40-ci-signature-guard.yml` (`Maint 40 CI Signature Guard`) | `push`/`pull_request` targeting `phase-2-dev` | Validates the signed job manifest for `pr-10-ci-python.yml`.
| `maint-90-selftest.yml` (`Maint 90 Selftest`) | Weekly cron, manual | Thin wrapper that dispatches the reusable CI self-test matrix.

### Agent automation entry points

| Workflow | Trigger(s) | Purpose |
|----------|------------|---------|
| `agents-43-codex-issue-bridge.yml` (`Agents 43 Codex Issue Bridge`) | `issues`, manual | Prepares Codex-ready branches/PRs when an `agent:codex` label is applied.
| `agents-70-orchestrator.yml` (`Agents 70 Orchestrator`) | 20-minute cron, manual | Unified agents toolkit (readiness probes, Codex bootstrap, watchdogs) delegating to `reusable-70-agents.yml`.

### Reusable composites

| Workflow | Consumed by | Notes |
|----------|-------------|-------|
| `reusable-70-agents.yml` (`Reusable 70 Agents`) | `agents-70-orchestrator.yml` | Implements readiness, bootstrap, diagnostics, and watchdog jobs.
| `reusable-90-ci-python.yml` (`Reusable 90 CI Python`) | `maint-90-selftest.yml` | Legacy matrix executor retained for self-tests while consumers migrate to the single-job workflow.
| `reusable-92-autofix.yml` (`Reusable 92 Autofix`) | `maint-32-autofix.yml` | Autofix harness invoked after CI gates finish.
| `reusable-94-legacy-ci-python.yml` (`Reusable 94 Legacy CI Python`) | Downstream consumers | Compatibility shim for repositories that still need the old matrix layout.
| `reusable-99-selftest.yml` (`Reusable 99 Selftest`) | `maint-90-selftest.yml` | Matrix smoke-test covering reusable CI feature combinations.

## Contributor Quick Start

Follow this sequence before pushing workflow changes or large code edits:

1. **Install tooling** – run `./scripts/setup_env.sh` once to create a virtual
   environment with repository requirements.
2. **Mirror the CI style gate locally** – execute:

   ```bash
   ./scripts/style_gate_local.sh
   ```

   The script sources `.github/workflows/autofix-versions.env`, installs the
   pinned formatter/type versions, runs Ruff/Black, and finishes with a mypy pass
   over `src/trend_analysis` and `src/trend_portfolio_app`. Fix any reported
   issues to keep `PR 10 CI Python` green.
3. **Targeted tests** – add `pytest tests/test_workflow_naming.py` after editing
   workflow files to ensure naming conventions hold. For agents changes, also run
   `pytest tests/test_automation_workflows.py -k agents`.
4. **Optional smoke** – `gh workflow list --limit 20` validates that only the
   documented workflows surface in the Actions tab.

## Adding or Renumbering Workflows

1. Pick the correct prefix/number band (see Naming Policy) and choose the lowest
   unused slot.
2. Place the workflow in `.github/workflows/` with the matching Title Case
   `name:`.
3. Update any trigger dependencies (`workflow_run` consumers) so maintenance jobs
   continue to listen to the correct producers.
4. Document the change in this file (inventory tables + bands) and in
   `docs/WORKFLOW_GUIDE.md` if the topology shifts.
5. Run the validation commands listed above before opening a PR.

## Formatter & Type Checker Pins

- `.github/workflows/autofix-versions.env` is the single source of truth for
  formatter/type tooling versions (Ruff, Black, isort, docformatter, mypy).
- `pr-10-ci-python.yml`, `reusable-90-ci-python.yml`, and the autofix composite
  action all load and validate this env file before installing tools; they fail
  fast if the file is missing or incomplete.
- Local mirrors (`scripts/style_gate_local.sh`, `scripts/dev_check.sh`,
  `scripts/validate_fast.sh`) source the same env file so contributors run the
  identical versions before pushing.
- When bumping any formatter, update the env file first, rerun
  `./scripts/style_gate_local.sh`, and let CI confirm the new version to keep
  automation and local flows aligned.

## CI Signature Guard Fixtures

`maint-40-ci-signature-guard.yml` enforces a manifest signature for the PR Python
workflow by comparing two fixture files stored in `.github/signature-fixtures/`:

- `basic_jobs.json` – canonical list of jobs (name, concurrency label, metadata)
  that must exist in `pr-10-ci-python.yml`.
- `basic_hash.txt` – precomputed hash of the JSON payload used by
  `.github/actions/signature-verify` to detect unauthorized job changes.

When intentionally editing CI jobs, regenerate `basic_jobs.json`, compute the new
hash, and update both files in the same commit. Use
`tools/test_failure_signature.py` locally to recompute and verify the hash before
pushing. The guard only runs on pushes/PRs targeting `phase-2-dev` and publishes
 a step summary linking back here.

## Agents `options_json` Schema

`agents-70-orchestrator.yml` accepts the standard dispatch inputs shown in the
workflow plus an extensible JSON payload routed through `options_json`. The JSON
is parsed with `fromJson()` and handed to the reusable agents workflow.

```jsonc
{
  "diagnostic_mode": "off" | "dry-run" | "full",
  "readiness_custom_logins": "login-a,login-b",
  "codex_command_phrase": "@codex start"
}
```

- **`diagnostic_mode`** — `off` (default) disables diagnostics, `dry-run` keeps
  bootstrap logic read-only, `full` allows branch creation and sets
  `diagnostic_attempt_branch=true` in the reusable workflow.
- **`readiness_custom_logins`** — optional comma-separated list of extra agent
  logins to probe during readiness checks.
- **`codex_command_phrase`** — overrides the PR comment phrase the orchestrator
  looks for when confirming Codex bootstrap completion.

Extend the schema by adding new keys to the JSON object and updating both the
orchestrator workflow and this documentation. Keep additional toggles in the
JSON payload to avoid exceeding GitHub's 10 input limit on `workflow_dispatch`
forms.

## Quick Validation Commands

- `pytest tests/test_workflow_naming.py` — guard the naming convention.
- `pytest tests/test_automation_workflows.py -k agents` — ensure agents inputs
  (including `options_json`) parse correctly.
- `gh workflow list --limit 20` — verify only the inventoried workflows are
  visible in the Actions UI.

Run the naming test after any workflow change to keep CI guardrails intact.
