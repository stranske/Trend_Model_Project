# Workflow Catalog & Contributor Quick Start

Use this page as the canonical reference for CI workflow naming, inventory, and
local guardrails. It consolidates the requirements from Issues #2190 and #2202
and adds guidance for contributors who need to stage new workflows or dispatch
the agents toolkit. The quick catalog below spells out **purpose, triggers,
required permissions, and status/label effects** for every merge gate and agent
automation entry point currently in service.

## CI & agents quick catalog

| Workflow | File | What it does | When it runs | Required secrets / permissions | Status outputs & labels |
| --- | --- | --- | --- | --- | --- |
| **Gate** | `.github/workflows/pr-gate.yml` | Fan-out orchestrator that reuses the Python CI matrix and Docker smoke reusable workflows, then enforces all downstream results. | `pull_request` (non-doc paths) and `workflow_dispatch`. | Default `GITHUB_TOKEN` (`contents: read`) for all jobs; delegated reusable workflows do not request additional scopes when called from Gate. | <ul><li>**Status checks:** `core tests (3.11)`, `core tests (3.12)`, `docker smoke`, aggregate `gate`.</li><li>**Labels:** _none_.</li></ul> |
| **Autofix** | `.github/workflows/autofix.yml` | Lightweight formatting/type hygiene runner that auto-commits fixes for same-repo PRs or uploads a patch for forks. | `pull_request` (including label changes). | `contents: write`, `pull-requests: write`; inherits repository secrets but does not require extra PATs. | <ul><li>**Status checks:** top-level `apply` job delegating to the `autofix` composite.</li><li>**Labels:** `autofix`, `autofix:applied`/`autofix:patch`, mutually exclusive `autofix:clean` vs `autofix:debt`.</li></ul> |
| **Repo Health (Maint 02)** | `.github/workflows/maint-02-repo-health.yml` | Weekly sweep that summarises stale branches and unassigned issues in the run summary. | Monday cron (`15 7 * * 1`) plus `workflow_dispatch`. | `contents: read`, `issues: read`; no secrets required. | <ul><li>**Status checks:** `Weekly repository health sweep`.</li><li>**Labels:** _none_.</li></ul> |
| **Maint 35 Repo Health Self Check** | `.github/workflows/maint-35-repo-health-self-check.yml` | Read-only repository health probe that reports label coverage and branch-protection visibility via the step summary. | Weekly cron (`20 6 * * 1`) and `workflow_dispatch`. | `contents: read`, `issues: read`, `pull-requests: read`, `actions: read`. | <ul><li>**Status checks:** none – informational summary only.</li><li>**Labels:** _none_.</li></ul> |
| **Agents Consumer** | `.github/workflows/agents-consumer.yml` | Hourly/adhoc entry point that parses JSON input then dispatches to the reusable agents toolkit for readiness, bootstrap, watchdog, and keepalive tasks. | Hourly cron (`15 * * * *`) and `workflow_dispatch` with `params_json`. | `contents`, `pull-requests`, `issues`: `write`; optional `service_bot_pat` forwarded to downstream jobs. | <ul><li>**Status checks:** `Resolve Parameters`, `Dispatch Agents Toolkit`, and whichever delegated runs are enabled (e.g., `Agent Readiness Probe`, `Codex Preflight`, `Bootstrap Codex PRs`, `Codex Keepalive Sweep`, `Agent Watchdog`).</li><li>**Labels:** Bootstrap runs add `agent:codex` to spawned PRs.</li></ul> |
| **Reuse Agents** | `.github/workflows/reuse-agents.yml` | Workflow-call wrapper so other repositories or orchestrators can invoke the agents toolkit with consistent inputs. | `workflow_call` only. | Same as Agents Consumer (`contents`, `pull-requests`, `issues`: `write`) and can accept a `service_bot_pat` secret for Codex bootstrap. | <ul><li>**Status checks:** Top-level `call` job plus the same delegated checks from `Reusable 70 Agents` (readiness, preflight, bootstrap, watchdog, keepalive) when requested.</li><li>**Labels:** Mirrors `codex-bootstrap-lite` (e.g., `agent:codex` for created PRs).</li></ul> |

## Naming Policy & Number Ranges

- Store workflows under `.github/workflows/` and follow the
  `<area>-<NN>-<slug>.yml` convention.
- Reserve number bands per area so future additions remain grouped:

  | Prefix | Number slots | Usage | Notes |
  |--------|--------------|-------|-------|
  | `pr-` | `10–19` | Pull-request gates | `pr-gate.yml` is the primary orchestrator; keep space for specialized PR jobs (docs, optional helpers).
  | `maint-` | `00–49` and `90s` | Scheduled/background maintenance | Low numbers for repo hygiene, 30s/40s for post-CI and guards, 90 for self-tests calling reusable matrices.
  | `agents-` | `40s` & `70s` | Agent bootstrap/orchestration | `43` bridges issues to Codex; `70` is the manual/scheduled orchestrator.
  | `reusable-` | `70s` & `90s` | Composite workflows invoked by others | Keep 90s for CI executors, 70s for agent composites.

- Match the `name:` field to the filename rendered in Title Case
  (`pr-gate.yml` → `Gate`).
- `tests/test_workflow_naming.py` enforces this policy—rerun it after modifying
  or adding workflows.
- When introducing a new workflow choose the lowest unused slot inside the
  appropriate band, update this document, and add the workflow to the relevant
  section below.

## Workflow Inventory

### Required PR gates (block merges by default)

| Workflow | Trigger(s) | Why it matters |
|----------|------------|----------------|
| `pr-gate.yml` (`Gate`) | `pull_request`, `workflow_dispatch` | Composite orchestrator that chains the reusable CI and Docker smoke jobs and enforces that every leg succeeds.
| `pr-14-docs-only.yml` (`PR 14 Docs Only`) | `pull_request` (doc paths) | Detects documentation-only diffs and posts a friendly skip notice instead of running heavier gates.
| `autofix.yml` (`Autofix`) | `pull_request` | Lightweight formatting/type-hygiene runner that auto-commits safe fixes or publishes a patch artifact for forked PRs.

**Operational details**
- **Gate** – Permissions: defaults (read scope). Secrets: relies on `GITHUB_TOKEN` only. Status outputs: `core tests (3.11)`, `core tests (3.12)`, `docker smoke`, and the aggregator job `gate`, which fails if any dependency fails.
- **Autofix** – Permissions: `contents: write`, `pull-requests: write`. Secrets: inherits `GITHUB_TOKEN` (sufficient for label + comment updates). Status outputs: `autofix` job; labels applied include `autofix`, `autofix:applied`/`autofix:patch`, and cleanliness toggles (`autofix:clean`/`autofix:debt`).

These jobs must stay green for PRs to merge. The post-CI maintenance jobs below
listen to their `workflow_run` events.

### Maintenance & observability (scheduled/optional reruns)

| Workflow | Trigger(s) | Purpose |
|----------|------------|---------|
| `maint-02-repo-health.yml` (`Maint 02 Repo Health`) | Weekly cron, manual | Reports stale branches & unassigned issues.
| `maint-post-ci.yml` (`Maint Post CI`) | `workflow_run` (Gate) | Consolidated follower that posts Gate summaries, applies low-risk autofix commits, and owns CI failure-tracker updates.
| `maint-33-check-failure-tracker.yml` (`Maint 33 Check Failure Tracker`) | `workflow_run` (Gate) | Lightweight compatibility shell that documents the delegation to `maint-post-ci.yml`.
| `maint-35-repo-health-self-check.yml` (`Maint 35 Repo Health Self Check`) | Weekly cron, manual | Read-only repo health pulse that surfaces missing labels or branch-protection visibility gaps in the step summary.
| `maint-36-actionlint.yml` (`Maint 36 Actionlint`) | `pull_request`, weekly cron, manual | Sole workflow-lint gate (actionlint via reviewdog).
| `maint-40-ci-signature-guard.yml` (`Maint 40 CI Signature Guard`) | `push`/`pull_request` targeting `phase-2-dev` | Validates the signed job manifest for `pr-gate.yml`.
| `maint-41-chatgpt-issue-sync.yml` (`Maint 41 ChatGPT Issue Sync`) | `workflow_dispatch` (manual) | Fans out curated topic lists (e.g. `Issues.txt`) into labeled GitHub issues. ⚠️ Repository policy: do not remove without a functionally equivalent replacement. |
| `maint-45-cosmetic-repair.yml` (`Maint 45 Cosmetic Repair`) | `workflow_dispatch` | Manual pytest + guardrail fixer that applies tolerance/snapshot updates and opens a labelled PR when drift is detected. |

**Operational details**
- **Maint 02 Repo Health** – Permissions: `contents: read`, `issues: read`. Secrets: uses `GITHUB_TOKEN` only. Output: publishes a step summary (“Repository health weekly sweep”) with stale branches and unassigned-issue tables.

### Agent automation entry points

| Workflow | Trigger(s) | Purpose |
|----------|------------|---------|
| `agents-consumer.yml` (`Agents Consumer`) | Hourly cron, manual | Consolidated wrapper that accepts a JSON payload and calls `reuse-agents.yml` to run readiness, preflight, verification, and bootstrap flows.
| `agents-43-codex-issue-bridge.yml` (`Agents 43 Codex Issue Bridge`) | `issues`, manual | Prepares Codex-ready branches/PRs when an `agent:codex` label is applied.
| `agents-44-verify-agent-assignment.yml` (`Agents 44 Verify Agent Assignment`) | `workflow_call`, manual | Checks that `agent:codex` issues remain assigned to an approved automation account before downstream workflows act.
| `agents-70-orchestrator.yml` (`Agents 70 Orchestrator`) | 20-minute cron, manual | Unified agents toolkit (readiness probes, Codex bootstrap, watchdogs) delegating to `reusable-70-agents.yml`.

**Operational details**
- **Agents Consumer** – Permissions: `contents: write`, `pull-requests: write`, `issues: write`. Secrets: inherits `GITHUB_TOKEN` and forwards `secrets.SERVICE_BOT_PAT` when available so downstream automation may push branches/comments. Output: emits `Resolve Parameters` summary and a `Dispatch Agents Toolkit` reusable call that surfaces Codex readiness/watchdog diagnostics.
- **Agents 70 Orchestrator** – Uses `options_json` to layer advanced toggles without adding dispatch inputs; set `enable_bootstrap: true` (and optionally `bootstrap_issues_label`) to fan bootstrap jobs through the reusable workflow during manual runs.
- The standalone `.github/workflows/agent-watchdog.yml` workflow has been removed; run watchdog checks by dispatching the orchestrator with `enable_watchdog: true` (default) or via the Agents Consumer/Reuse wrappers.

### Reusable composites

| Workflow | Consumed by | Notes |
|----------|-------------|-------|
| `reuse-agents.yml` (`Reuse Agents`) | `agents-consumer.yml` | Bridges `params_json` inputs to the reusable toolkit while preserving defaults.
| `reusable-70-agents.yml` (`Reusable 70 Agents`) | `agents-70-orchestrator.yml`, `reuse-agents.yml` | Implements readiness, bootstrap, diagnostics, and watchdog jobs.
| `reusable-92-autofix.yml` (`Reusable 92 Autofix`) | `maint-post-ci.yml`, `autofix.yml` | Autofix harness used both by the PR-time autofix workflow and the post-CI maintenance listener.
| `reusable-99-selftest.yml` (`Reusable 99 Selftest`) | `maint-` self-test orchestration | Scenario matrix that validates the reusable CI executor and artifact inventory.
| `reusable-ci.yml` (`Reusable CI`) | Gate, downstream repositories | Single source for Python lint/type/test coverage runs.
| `reusable-docker.yml` (`Reusable Docker Smoke`) | Gate, downstream repositories | Docker build + smoke reusable consumed by Gate and external callers.

**Operational details**
- **Reuse Agents** – Permissions: `contents: write`, `pull-requests: write`, `issues: write`. Secrets: optional `service_bot_pat` (forwarded to `reusable-70-agents`) plus `GITHUB_TOKEN`. Outputs: single `call` job exposes reusable outputs such as `triggered` keepalive list and watchdog diagnostics for upstream orchestrators.

### Archived self-test workflows

`Old/workflows/maint-90-selftest.yml` remains available as the historical wrapper that previously
scheduled the self-test cron. The retired PR comment and maintenance wrappers listed below stay
removed; consult git history if you need their YAML for archaeology:

- `maint-43-selftest-pr-comment.yml` – deleted; previously posted PR comments summarising self-test matrices.
- `maint-44-selftest-reusable-ci.yml` – deleted reusable-integration cron (matrix coverage moved to the
  main CI jobs).
- `maint-48-selftest-reusable-ci.yml` – deleted; short-lived variant of the reusable matrix exerciser.
- `pr-20-selftest-pr-comment.yml` – deleted; PR-triggered comment bot that duplicated the maintenance
  wrapper.

See [ARCHIVE_WORKFLOWS.md](../../ARCHIVE_WORKFLOWS.md) for the full ledger of retired workflows and
rationale, including notes on the removed files.

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
   issues to keep the Gate workflow green.
3. **Targeted tests** – add `pytest tests/test_workflow_naming.py` after editing
   workflow files to ensure naming conventions hold. For agents changes, also run
   `pytest tests/test_automation_workflows.py -k agents`.
4. **Optional smoke** – `gh workflow list --limit 20` validates that only the
   documented workflows surface in the Actions tab.

## Adding or Renumbering Workflows

1. Pick the correct prefix/number band (see Naming Policy) and choose the lowest
   unused slot.
   - Treat the `NN` portion as a zero-padded two-digit identifier within the
     band (`pr-10`, `maint-36`, etc.). Check the tables above before reusing a
     number so future contributors can infer gaps at a glance.
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
- `reusable-ci.yml`, `reusable-docker.yml`, and the autofix composite
  action all load and validate this env file before installing tools; they fail
  fast if the file is missing or incomplete.
- Local mirrors (`scripts/style_gate_local.sh`, `scripts/dev_check.sh`,
  `scripts/validate_fast.sh`) source the same env file so contributors run the
  identical versions before pushing.
- When bumping any formatter, update the env file first, rerun
  `./scripts/style_gate_local.sh`, and let CI confirm the new version to keep
  automation and local flows aligned.

## CI Signature Guard Fixtures

`maint-40-ci-signature-guard.yml` enforces a manifest signature for the Gate
workflow by comparing two fixture files stored in `.github/signature-fixtures/`:

- `basic_jobs.json` – canonical list of jobs (name, concurrency label, metadata)
  that must exist in `pr-gate.yml`.
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
