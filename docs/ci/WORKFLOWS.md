# Workflow Catalog & Contributor Quick Start

Use this page as the canonical reference for CI workflow naming, inventory, and
local guardrails. It consolidates the requirements from Issues #2190, #2202, and
#2466. The Gate workflow remains the required merge check for every pull
request. Agents automation now exposes a single entry point: the scheduled and
manually dispatchable **Agents 70 Orchestrator**. Deprecated compatibility
wrappers are listed later in this guide for historical context only.

## CI & agents quick catalog

Use the matrix below as the authoritative roster of active workflows. Each row captures the canonical triggers, permission scopes, and whether the workflow blocks merges (`Required?`). Reusable composites appear at the end because they expose only `workflow_call` entry points.

| Workflow | File | Trigger(s) | Permissions | Required? | Purpose |
| --- | --- | --- | --- | --- | --- |
| **Gate** | `.github/workflows/pr-00-gate.yml` | `pull_request` (non-doc paths), `workflow_dispatch` | Defaults (`contents: read`) via `GITHUB_TOKEN`; delegated jobs reuse the caller token. | **Yes** – aggregate `gate` status must pass. | Fan-out orchestrator chaining the reusable Python CI and Docker smoke jobs; enforces downstream results. |
| **PR 14 Docs Only** | `.github/workflows/pr-14-docs-only.yml` | `pull_request` (doc-only diffs) | `contents: read` | **Conditional** – posts skip notice and exits success. | Detects doc-only PRs and short-circuits heavier CI. |
| **Autofix** | `.github/workflows/autofix.yml` | `pull_request` (including label updates) | `contents: write`, `pull-requests: write` | **Yes** – `apply` job must succeed. | Runs the reusable autofix composite to apply/offer safe formatting fixes. |
| **Maint Post CI** | `.github/workflows/maint-30-post-ci.yml` | `workflow_run` (Gate), `workflow_dispatch` | `contents: write`, `pull-requests: write`, `issues: write` | No | Consolidated follower that posts Gate summaries, repeats autofix, and updates the `ci-failure` rollup issue. |
| **Maint 02 Repo Health** | `.github/workflows/health-41-repo-health.yml` | Monday cron (`15 7 * * 1`), `workflow_dispatch` | `contents: read`, `issues: read` | No | Weekly stale-branch and unassigned-issue sweep. |
| **Maint 33 Check Failure Tracker** | `.github/workflows/maint-33-check-failure-tracker.yml` | `workflow_run` (Gate) | `contents: read` | No | Compatibility shell documenting delegation to Maint Post CI. |
| **Maint 35 Repo Health Self Check** | `.github/workflows/health-40-repo-selfcheck.yml` | Weekly cron (`20 6 * * 1`), `workflow_dispatch` | `contents: read`, `issues: read`, `pull-requests: read`, `actions: read` | No | Read-only probe that reports label coverage and branch-protection visibility. |
| **Maint 36 Actionlint** | `.github/workflows/health-42-actionlint.yml` | `pull_request`, weekly cron, `workflow_dispatch` | `contents: read` | No | Workflow-lint gate using `actionlint` via reviewdog. |
| **Maint 40 CI Signature Guard** | `.github/workflows/health-43-ci-signature-guard.yml` | `push`/`pull_request` targeting `phase-2-dev` | `contents: read` | No | Validates the signed job manifest for Gate. |
| **Maint 41 ChatGPT Issue Sync** | `.github/workflows/agents-63-chatgpt-issue-sync.yml` | `workflow_dispatch` | `contents: write`, `issues: write` | No | Manual sync that turns curated topic lists (e.g. `Issues.txt`) into labeled GitHub issues. |
| **Maint 45 Cosmetic Repair** | `.github/workflows/maint-34-cosmetic-repair.yml` | `workflow_dispatch` | `contents: write`, `pull-requests: write` | No | Manual pytest + guardrail fixer that opens a labeled PR when drift is detected. |
| **Enforce Gate Branch Protection** | `.github/workflows/health-44-gate-branch-protection.yml` | Cron (`0 6 * * *`), `workflow_dispatch` | `contents: read`, `pull-requests: read`; optional `BRANCH_PROTECTION_TOKEN` | No | Validates branch protection settings via helper script; no-ops if PAT absent. |
| **Agents 43 Codex Issue Bridge** | `.github/workflows/agents-43-codex-issue-bridge.yml` | `issues`, `workflow_dispatch` | `contents: write`, `pull-requests: write`, `issues: write`; optional `service_bot_pat` | No | Label-driven helper that prepares Codex bootstrap issues/PRs and optionally comments `@codex start`. |
| **Agents 70 Orchestrator** | `.github/workflows/agents-70-orchestrator.yml` | Cron (`*/20 * * * *`), `workflow_dispatch` | `contents: write`, `pull-requests: write`, `issues: write`; optional `service_bot_pat` | No | Primary automation entry point dispatching readiness, bootstrap, diagnostics, and keepalive routines. |
| **Agents 44 Verify Agent Assignment** | `.github/workflows/agents-64-verify-agent-assignment.yml` | `workflow_call`, `workflow_dispatch` | `issues: read` | No | Reusable issue-verification helper used by the orchestrator and available for ad-hoc checks. |
| **Reusable CI** | `.github/workflows/reusable-10-ci-python.yml` | `workflow_call` | Inherits caller permissions | No | Python lint/type/test reusable consumed by Gate and downstream repositories. |
| **Reusable Docker Smoke** | `.github/workflows/reusable-12-ci-docker.yml` | `workflow_call` | Inherits caller permissions | No | Docker build + smoke reusable consumed by Gate and external callers. |
| **Reusable 92 Autofix** | `.github/workflows/reusable-92-autofix.yml` | `workflow_call` | `contents: write`, `pull-requests: write` | No | Autofix harness shared by `autofix.yml` and `maint-post-ci.yml`. |
| **Reusable 70 Agents** | `.github/workflows/reusable-70-agents.yml` | `workflow_call` | `contents: write`, `pull-requests: write`, `issues: write`; optional `service_bot_pat` | No | Sole agents composite implementing readiness, bootstrap, diagnostics, keepalive, and watchdog jobs for all callers (orchestrator, manual consumers, downstream repos). |
| **Reusable 99 Selftest** | `.github/workflows/reusable-99-selftest.yml` | `workflow_call` | `contents: read` | No | Scenario matrix validating the reusable CI executor. |

## Naming Policy & Number Ranges

- Store workflows under `.github/workflows/` and follow the
  `<area>-<NN>-<slug>.yml` convention.
- Reserve number bands per area so future additions remain grouped:

  | Prefix | Number slots | Usage | Notes |
  |--------|--------------|-------|-------|
  | `pr-` | `10–19` | Pull-request gates | `pr-00-gate.yml` is the primary orchestrator; keep space for specialized PR jobs (docs, optional helpers).
  | `maint-` | `00–49` and `90s` | Scheduled/background maintenance | Low numbers for repo hygiene, 30s/40s for post-CI and guards, 90 for self-tests calling reusable matrices.
  | `agents-` | `70s` | Agent bootstrap/orchestration | `agents-70-orchestrator.yml` handles automation cadences; legacy consumers remain manual-only for compatibility.
  | `reusable-` | `70s` & `90s` | Composite workflows invoked by others | Keep 90s for CI executors, 70s for agent composites.

- Match the `name:` field to the filename rendered in Title Case
  (`pr-00-gate.yml` → `Gate`).
- `tests/test_workflow_naming.py` enforces this policy—rerun it after modifying
  or adding workflows.
- When introducing a new workflow choose the lowest unused slot inside the
  appropriate band, update this document, and add the workflow to the relevant
  section below.

## Workflow Inventory

### Required PR gates (block merges by default)

| Workflow | Trigger(s) | Why it matters |
|----------|------------|----------------|
| `pr-00-gate.yml` (`Gate`) | `pull_request`, `workflow_dispatch` | Composite orchestrator that chains the reusable CI and Docker smoke jobs and enforces that every leg succeeds.
| `pr-14-docs-only.yml` (`PR 14 Docs Only`) | `pull_request` (doc paths) | Detects documentation-only diffs and posts a friendly skip notice instead of running heavier gates.
| `autofix.yml` (`Autofix`) | `pull_request` | Lightweight formatting/type-hygiene runner that auto-commits safe fixes or publishes a patch artifact for forked PRs.

**Operational details**
- **Gate** – Permissions: defaults (read scope). Secrets: relies on `GITHUB_TOKEN` only. Status outputs: `core tests (3.11)`, `core tests (3.12)`, `docker smoke`, and the aggregator job `gate`, which fails if any dependency fails.
- **Autofix** – Permissions: `contents: write`, `pull-requests: write`. Secrets: inherits `GITHUB_TOKEN` (sufficient for label + comment updates). Status outputs: `apply` job; labels applied include `autofix`, `autofix:applied`/`autofix:patch`, and cleanliness toggles (`autofix:clean`/`autofix:debt`).

**Gate job map**

| Job | Role | Artifacts / Outputs |
| --- | ---- | ------------------- |
| `core-tests-311` | Python 3.11 lint/type/test run via `reusable-10-ci-python.yml`. | Uploads `coverage-3.11` artifact (coverage XML + HTML summary). |
| `core-tests-312` | Python 3.12 lint/type/test run via `reusable-10-ci-python.yml`. | Uploads `coverage-3.12` artifact mirroring 3.11 contents. |
| `docker-smoke` | Container build + smoke test via `reusable-12-ci-docker.yml`. | No artifacts; emits build + smoke logs in job summary. |
| `gate` | Final enforcement job that downloads coverage artifacts, posts the summary table, and fails if any upstream job failed. | Adds `Gate status` table to the workflow summary and surfaces missing artifacts in the log. |

These jobs must stay green for PRs to merge. The post-CI maintenance jobs below listen to their `workflow_run` events and post summaries whenever the Gate aggregator fails.

### Maintenance & observability (scheduled/optional reruns)

| Workflow | Trigger(s) | Purpose |
|----------|------------|---------|
| `health-41-repo-health.yml` (`Maint 02 Repo Health`) | Weekly cron, manual | Reports stale branches & unassigned issues.
| `maint-30-post-ci.yml` (`Maint Post CI`) | `workflow_run` (Gate) | Consolidated follower that posts Gate summaries, applies low-risk autofix commits, and owns CI failure-tracker updates.
| `maint-33-check-failure-tracker.yml` (`Maint 33 Check Failure Tracker`) | `workflow_run` (Gate) | Lightweight compatibility shell that documents the delegation to `maint-30-post-ci.yml`.
| `health-40-repo-selfcheck.yml` (`Maint 35 Repo Health Self Check`) | Weekly cron, manual | Read-only repo health pulse that surfaces missing labels or branch-protection visibility gaps in the step summary.
| `health-44-gate-branch-protection.yml` (`Enforce Gate Branch Protection`) | Hourly cron, manual | Applies branch-protection policy checks using `tools/enforce_gate_branch_protection.py`; skips gracefully when the PAT is not configured.
| `health-42-actionlint.yml` (`Maint 36 Actionlint`) | `pull_request`, weekly cron, manual | Sole workflow-lint gate (actionlint via reviewdog).
| `health-43-ci-signature-guard.yml` (`Maint 40 CI Signature Guard`) | `push`/`pull_request` targeting `phase-2-dev` | Validates the signed job manifest for `pr-00-gate.yml`.
| `agents-63-chatgpt-issue-sync.yml` (`Maint 41 ChatGPT Issue Sync`) | `workflow_dispatch` (manual) | Fans out curated topic lists (e.g. `Issues.txt`) into labeled GitHub issues. ⚠️ Repository policy: do not remove without a functionally equivalent replacement. |
| `maint-34-cosmetic-repair.yml` (`Maint 45 Cosmetic Repair`) | `workflow_dispatch` | Manual pytest + guardrail fixer that applies tolerance/snapshot updates and opens a labelled PR when drift is detected. |

**Repo health troubleshooting**
- If `Maint 02 Repo Health` fails with `Resource not accessible by integration`, the `GITHUB_TOKEN` lacks `issues: read` scope. Update the workflow's fine-grained token permissions (Repository → Settings → Actions → General) to grant `issues: read`, or rerun manually with a PAT stored in `SERVICE_BOT_PAT` that includes `repo` read scopes.
- Confirm the repository allows GitHub Actions to read all branches. Private forks that disable default token scopes can also trigger the same error; re-enable the default permissions or provide an explicit PAT.

### CI failure rollup issue

`Maint Post CI` maintains **one** open issue labelled `ci-failure` that aggregates "CI failures in the last 24 h". The failure-tracker job updates the table in place with each Gate failure, links the offending workflow run, and closes the issue automatically once the inactivity threshold elapses. The issue carries labels `ci`, `devops`, and `priority: medium`; escalations add `priority: high` when the same signature trips three times. Use this issue for a quick dashboard of outstanding CI problems instead of scanning individual PR timelines.

### Agent automation entry points

`agents-70-orchestrator.yml` (`Agents 70 Orchestrator`) is the scheduled automation entry point. It runs on an hourly cron and can also be dispatched manually. Both methods call the reusable agents toolkit to perform readiness probes, Codex bootstrap, diagnostics, and keepalive sweeps.
Deprecated compatibility wrappers (`agents-62-consumer.yml`, `agents-consumer.yml`) remain available for curated manual runs that require the historical JSON schema; they are not part of the supported flow and should be referenced only when migrating older automations.
The Codex Issue Bridge is only a label-driven helper for seeding bootstrap PRs. `agents-64-verify-agent-assignment.yml` exposes the verification logic as a reusable workflow-call entry point.

**Operational details**
- Provide required write scopes via the default `GITHUB_TOKEN`. Supply `service_bot_pat` when bootstrap jobs must push branches or leave comments.
- Use the `params_json` payload to enable bootstrap (`{"enable_bootstrap": true}`) or pass extra labels such as `{"bootstrap_issues_label": "agent:codex"}` when dispatching manually. The orchestrator parses the JSON via `fromJson()` and forwards toggles to `reusable-70-agents.yml`.
- Readiness, preflight, bootstrap, and keepalive diagnostics appear in the job summary. Failures bubble up through the single `orchestrate` job; Maint Post CI will echo the failing run link in the CI failure-tracker issue when the Gate is affected.

### Manual Orchestrator Dispatch

1. Navigate to **Actions → Agents 70 Orchestrator → Run workflow**.
2. Provide inputs:
   - **Branch**: default (`phase-2-dev`) unless testing a feature branch.
   - Flip the toggle inputs as needed (`enable_readiness`, `enable_preflight`, `enable_watchdog`, etc.).
   - **Params JSON**: bundle less common overrides (custom logins, diagnostics, bootstrap switches) inside the JSON payload. Example:
     ```json
     {
       "enable_bootstrap": true,
       "bootstrap_issues_label": "agent:codex",
       "diagnostic_mode": "dry-run",
       "readiness_custom_logins": "backup-bot",
       "options_json": "{\"enable_keepalive\":false,\"keepalive\":{\"enabled\":false}}"
     }
     ```
3. Click **Run workflow**. The orchestrator calls `reusable-70-agents.yml`; job summaries include readiness tables, bootstrap status, and links to spawned PRs.
4. CLI dispatch example (same `params_json` schema):
   ```bash
   gh workflow run agents-70-orchestrator.yml \
     --ref phase-2-dev \
     --raw-field enable_watchdog=true \
     --raw-field params_json='{"enable_bootstrap":true,"bootstrap_issues_label":"agent:codex","diagnostic_mode":"dry-run"}'
   ```
   The CLI invocation mirrors the manual form—`params_json` is optional but provides a single string for complex settings.

### Deprecated manual wrappers

`agents-62-consumer.yml` and `agents-consumer.yml` remain in the repository for callers pinned to their historical interfaces. Both are manual-only, forward to `reusable-70-agents.yml`, and accept the same `params_json` payload documented above. Prefer the orchestrator for all new automation.

### Agent troubleshooting: bootstrap & readiness signals

| Symptom | Likely cause | Where to look | Remedy |
| ------- | ------------ | ------------- | ------ |
| Readiness probe fails immediately | Missing PAT or permissions | `orchestrate` job summary → “Authentication” step | Provide `SERVICE_BOT_PAT` secret or rerun with reduced scope. |
| Bootstrap skipped despite `enable_bootstrap` | No matching labelled issues | Job summary → “Bootstrap Planner” table | Add `agent:codex` label (or configured label) to target issues, rerun. |
| Bootstrap run exits with “Repository dirty” | Prior automation left branches open | Job log → `cleanup` step | Manually close stale branches or enable cleanup via the `params_json` payload (set `{"options_json":"{\\"cleanup\\":{\\"force\\":true}}"}`). |
| Readiness succeeds but Codex PR creation fails | Repository protections blocking pushes | Job log → `Create bootstrap branch` step | Ensure branch protection rules allow the automation account or supply a PAT with required scopes. |

Escalate persistent failures by linking the failing run URL in the CI failure-tracker issue managed by Maint Post CI.

### Gate pipeline overview

| Stage | Job ID (Gate) | Failure visibility |
|-------|---------------|--------------------|
| Python quality (3.11) | `core tests (3.11)` | Surfaces Ruff, mypy, pytest failures in the job log and step summary with direct links to offending tests. |
| Python quality (3.12) | `core tests (3.12)` | Mirrors the 3.11 leg; investigate parity regressions here first. |
| Docker smoke | `docker smoke` | Logs Docker build output and runtime smoke tests; summary links point to failing commands. |
| Aggregator | `gate` | Fails if any upstream job fails and posts a consolidated summary link. Maint Post CI consumes this status to update the CI failure-tracker issue. |

**Post-change monitoring.** When agent workflows change:

- Tag the source issue with `ci-failure` so it stays visible during the observation window.
- Coordinate a 48-hour watch to confirm no scheduled or issue-triggered `agents-62-consumer` runs fire (manual dispatch is the only allowed path).
- Capture a brief note or screenshot of the clean Actions history before removing the tag and closing the issue.

Manual-only status means maintainers should review the Actions list during that window to ensure the retired cron trigger stays inactive.

### Reusable composites

| Workflow | Consumed by | Notes |
|----------|-------------|-------|
| `reusable-70-agents.yml` (`Reusable 70 Agents`) | `agents-70-orchestrator.yml`, `agents-62-consumer.yml`, downstream repositories | Implements readiness, bootstrap, diagnostics, keepalive, and watchdog jobs.
| `reusable-92-autofix.yml` (`Reusable 92 Autofix`) | `maint-30-post-ci.yml`, `autofix.yml` | Autofix harness used both by the PR-time autofix workflow and the post-CI maintenance listener.
| `reusable-10-ci-python.yml` (`Reusable CI`) | Gate, downstream repositories | Single source for Python lint/type/test coverage runs.
| `reusable-12-ci-docker.yml` (`Reusable Docker Smoke`) | Gate, downstream repositories | Docker build + smoke reusable consumed by Gate and external callers.

**Operational details**
- **Reusable 70 Agents** – Permissions: `contents: write`, `pull-requests: write`, `issues: write`. Secrets: optional `service_bot_pat` (forwarded to downstream jobs) plus `GITHUB_TOKEN`. Outputs: per-job readiness tables, bootstrap activity summaries, keepalive sweep details, and watchdog notes surfaced via job summaries and declared workflow outputs.

### Manual self-test examples

| Workflow | Notes |
|----------|-------|
| `selftest-81-reusable-ci.yml` (`Selftest 81 Reusable CI`) | Manual-only dispatch that runs the reusable CI matrix across the documented feature toggles and uploads verification summaries/artifacts for humans to inspect. |

> Self-test workflows are intended as reference exercises for maintainers. They are quiet by default—run them via `workflow_dispatch` when you need a fresh artifact inventory check or to validate reusable CI changes, and expect no automated runs in Actions history.

### Archived self-test workflows

`Old/workflows/maint-90-selftest.yml` remains available as the historical wrapper that previously scheduled the self-test cron. The retired PR comment and maintenance wrappers listed below stay removed; consult git history if you need their YAML for archaeology:

- `selftest-83-pr-comment.yml` – deleted; previously posted PR comments summarising self-test matrices.
- `selftest-84-reusable-ci.yml` – deleted reusable-integration cron (matrix coverage moved to the
  main CI jobs).
- `selftest-88-reusable-ci.yml` – deleted; short-lived variant of the reusable matrix exerciser.
- `selftest-82-pr-comment.yml` – deleted; PR-triggered comment bot that duplicated the maintenance
  wrapper.

See [ARCHIVE_WORKFLOWS.md](../../ARCHIVE_WORKFLOWS.md) for the full ledger of retired workflows and rationale, including notes on the removed files.

## Contributor Quick Start

Follow this sequence before pushing workflow changes or large code edits:

1. **Install tooling** – run `./scripts/setup_env.sh` once to create a virtual environment with repository requirements.
2. **Mirror the CI style gate locally** – execute:

   ```bash
   ./scripts/style_gate_local.sh
   ```

   The script sources `.github/workflows/autofix-versions.env`, installs the pinned formatter/type versions, runs Ruff/Black, and finishes with a mypy pass over `src/trend_analysis` and `src/trend_portfolio_app`. Fix any reported issues to keep the Gate workflow green.
3. **Targeted tests** – add `pytest tests/test_workflow_naming.py` after editing workflow files to ensure naming conventions hold. For agents changes, also run `pytest tests/test_automation_workflows.py -k agents`.
4. **Optional smoke** – `gh workflow list --limit 20` validates that only the documented workflows surface in the Actions tab.

## Adding or Renumbering Workflows

1. Pick the correct prefix/number band (see Naming Policy) and choose the lowest unused slot.
   - Treat the `NN` portion as a zero-padded two-digit identifier within the band (`pr-10`, `maint-36`, etc.). Check the tables above before reusing a number so future contributors can infer gaps at a glance.
2. Place the workflow in `.github/workflows/` with the matching Title Case `name:`.
3. Update any trigger dependencies (`workflow_run` consumers) so maintenance jobs continue to listen to the correct producers.
4. Document the change in this file (inventory tables + bands) and in `docs/WORKFLOW_GUIDE.md` if the topology shifts.
5. Run the validation commands listed above before opening a PR.

## Formatter & Type Checker Pins

- `.github/workflows/autofix-versions.env` is the single source of truth for
  formatter/type tooling versions (Ruff, Black, isort, docformatter, mypy).
- `reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`, and the autofix composite
  action all load and validate this env file before installing tools; they fail
  fast if the file is missing or incomplete.
- Local mirrors (`scripts/style_gate_local.sh`, `scripts/dev_check.sh`,
  `scripts/validate_fast.sh`) source the same env file so contributors run the
  identical versions before pushing.
- When bumping any formatter, update the env file first, rerun
  `./scripts/style_gate_local.sh`, and let CI confirm the new version to keep
  automation and local flows aligned.

## CI Signature Guard Fixtures

`health-43-ci-signature-guard.yml` enforces a manifest signature for the Gate workflow by comparing two fixture files stored in `.github/signature-fixtures/`:

- `basic_jobs.json` – canonical list of jobs (name, concurrency label, metadata)
  that must exist in `pr-00-gate.yml`.
- `basic_hash.txt` – precomputed hash of the JSON payload used by
  `.github/actions/signature-verify` to detect unauthorized job changes.

When intentionally editing CI jobs, regenerate `basic_jobs.json`, compute the new hash, and update both files in the same commit. Use `tools/test_failure_signature.py` locally to recompute and verify the hash before pushing. The guard only runs on pushes/PRs targeting `phase-2-dev` and publishes a step summary linking back here.

## Agents `params_json` Schema

`agents-70-orchestrator.yml` accepts the standard dispatch inputs shown in the workflow plus an extensible JSON payload routed through `params_json`. The JSON is parsed with `fromJson()` and handed to the reusable agents workflow. Nested automation knobs can be forwarded via the optional `options_json` string when the reusable composite expects more structured data.

```jsonc
{
  "enable_bootstrap": true,
  "bootstrap_issues_label": "agent:codex",
  "diagnostic_mode": "dry-run",
  "readiness_custom_logins": "login-a,login-b",
  "codex_command_phrase": "@codex start",
  "options_json": "{\"enable_keepalive\":false}"
}
```

- **`enable_bootstrap` / `bootstrap_issues_label`** — toggle Codex bootstrap and override the label scanned for candidate issues.
- **`diagnostic_mode`** — `off` (default) disables diagnostics, `dry-run` keeps bootstrap logic read-only, `full` allows branch creation and sets `draft_pr: false` when Codex is seeded.
- **`readiness_custom_logins`** — comma-separated list for additional readiness probes.
- **`codex_command_phrase`** — phrase used when the orchestrator comments on issues or PRs to summon Codex.
- **`options_json`** — pass-through blob for nested settings such as keepalive thresholds or cleanup toggles (see Gate troubleshooting table above).

Keep this schema backward compatible; add new keys sparingly and document them here when introduced.
