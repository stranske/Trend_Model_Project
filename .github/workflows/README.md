# Workflow & Agent Automation Quick Start (Issue #2466)

This guide gives maintainers a fast reference for the streamlined CI and agent
automation stack. Pair it with
[docs/WORKFLOW_GUIDE.md](../../docs/WORKFLOW_GUIDE.md) for the canonical
inventory and naming rules.

---
## 1. Architecture Snapshot

Core layers after the Issue #2466 consolidation:

- **Gate orchestrator (`pr-gate.yml`)** – single required check that fans out to
  the Python 3.11/3.12 jobs and the Docker smoke reusable, then reports the
  aggregate `Gate / gate` status.
- **Maint Post CI (`maint-post-ci.yml`)** – `workflow_run` follower that posts a
  consolidated summary comment, retries autofix via the reusable composite, and
  maintains the rolling `ci-failure` dashboard issue.
- **Agents 70 Orchestrator (`agents-70-orchestrator.yml`)** – the only scheduled
  or manual entry point for agent automation. It dispatches readiness probes,
  diagnostics, Codex bootstrap, keepalive nudges, and watchdog sweeps through
  the reusable agents toolkit.
- **Governance & health** – scheduled jobs (`maint-02`, `maint-35`,
  `maint-36`, `maint-40`, `maint-41`, `maint-45`) keep labels, workflow linting,
  and hygiene reports current.

### 1.1 Current CI Topology

| Lane | Workflow(s) | Purpose | Required Status |
|------|-------------|---------|-----------------|
| Gate orchestrator | `pr-gate.yml` (`gate`) | Coordinates the reusable Python and Docker jobs, failing fast when any leg fails. | Required (`Gate / gate`) |
| Docs-only detector | `pr-14-docs-only.yml` | Posts a skip notice for documentation-only diffs and exits success. | Conditional |
| Autofix lane | `autofix.yml` | PR autofix runner that applies safe formatting fixes via the reusable composite. | Required (`Autofix / apply`) |
| Post-CI follower | `maint-post-ci.yml` | Summarises Gate runs, mirrors the autofix sweep, and updates the `ci-failure` rollup issue. | Optional |

### 1.2 Naming Policy Snapshot

- Workflows under `.github/workflows/` follow the `<area>-<NN>-<slug>.yml`
  convention. Guard tests in `tests/test_workflow_naming.py` enforce the rule.
- Legacy wrappers live in `Old/workflows/` for archaeology. Consult
  [ARCHIVE_WORKFLOWS.md](../../ARCHIVE_WORKFLOWS.md) when you need a retired
  slug.
- Update this README and `WORKFLOW_AUDIT_TEMP.md` whenever workflows are added,
  renamed, or retired.

High-level flow:
1. PR opened → labelers tag paths and agents → Gate + Autofix run automatically.
2. Maint Post CI posts the consolidated result and updates the `ci-failure`
   tracker when Gate fails.
3. Agents 70 Orchestrator handles all automation (scheduled + manual) via the
   reusable agents composite.

---
## 2. Label Cheat Sheet

| Label | Purpose | Source |
|-------|---------|--------|
| `agent:codex` / `agent:copilot` | Marks automation-owned issues and PRs | Agent labeler |
| `from:codex` / `from:copilot` | Origin marker for automation PRs | Agent labeler |
| `autofix` / `autofix:applied` | Track PR autofix results | Autofix workflow |
| `ci-failure` | Pins the rolling CI dashboard issue | Maint Post CI |
| Area labels | Scope classification for review routing | Path labeler |

---
## 3. Required Secrets & Variables

| Name | Type | Req | Purpose | Notes |
|------|------|-----|---------|-------|
| `SERVICE_BOT_PAT` | Secret | Rec | Allows automation to push branches and leave comments | `repo` scope |
| `AUTOFIX_OPT_IN_LABEL` | Var | Opt | Overrides the default autofix opt-in label | Defaults internal |
| `OPS_HEALTH_ISSUE` | Var | Req | Issue number for repo-health updates | Repo health jobs skip updates when unset |

All other jobs rely on the default `GITHUB_TOKEN` permissions noted in the
workflow files.

---
## 4. Trigger Matrix

| Workflow | Trigger(s) | Notes |
|----------|------------|-------|
| `pr-gate.yml` | `pull_request`, `workflow_dispatch` | Required gate orchestrating Python (3.11/3.12) and Docker reusable workflows. |
| `pr-14-docs-only.yml` | `pull_request` (doc paths) | Detects doc-only diffs and posts a skip notice instead of running heavy CI. |
| `autofix.yml` | `pull_request` | Runs the reusable autofix composite; required `apply` job must succeed. |
| `maint-post-ci.yml` | `workflow_run` (Gate), `workflow_dispatch` | Consolidated follower that posts status comments and maintains the `ci-failure` rollup issue. |
| `maint-02-repo-health.yml` | Weekly cron, `workflow_dispatch` | Hygiene sweep summarising stale branches and unassigned issues. |
| `maint-33-check-failure-tracker.yml` | `workflow_run` (Gate) | Compatibility shell that documents delegation to Maint Post CI. |
| `maint-35-repo-health-self-check.yml` | Weekly cron, `workflow_dispatch` | Governance probe that reports label coverage and branch-protection visibility. |
| `maint-36-actionlint.yml` | `pull_request`, weekly cron, `workflow_dispatch` | Sole workflow-lint gate (actionlint via reviewdog). |
| `maint-40-ci-signature-guard.yml` | `pull_request`/`push` (`phase-2-dev`) | Verifies the signed Gate manifest. |
| `maint-41-chatgpt-issue-sync.yml` | `workflow_dispatch` | Manual sync turning curated lists into labelled issues. |
| `maint-45-cosmetic-repair.yml` | `workflow_dispatch` | Manual pytest + cosmetic fixer that opens a labelled PR when drift is detected. |
| `agents-70-orchestrator.yml` | Cron (`*/20 * * * *`), `workflow_dispatch` | Sole automation entry point calling the reusable agents toolkit. |
| `reusable-70-agents.yml` | `workflow_call` | Implements readiness, bootstrap, diagnostics, keepalive, and watchdog jobs. |
| `reuse-agents.yml` | `workflow_call` | Bridges external callers to the reusable agents stack with consistent defaults. |
| `reusable-ci.yml` | `workflow_call` | Python lint/type/test reusable consumed by Gate and downstream repositories. |
| `reusable-docker.yml` | `workflow_call` | Docker build + smoke reusable consumed by Gate and external callers. |
| `reusable-92-autofix.yml` | `workflow_call` | Autofix composite used by `autofix.yml` and `maint-post-ci.yml`. |
| `reusable-99-selftest.yml` | `workflow_call` | Scenario matrix validating the reusable CI executor. |

---
## 5. Adopt Reusable Workflows

CI consumer example:

```yaml
name: CI
on:
  workflow_call:
    inputs:
      marker:
        type: string
        default: "not quarantine and not slow"
      python-version:
        type: string
        default: "3.12"
jobs:
  ci:
    uses: stranske/Trend_Model_Project/.github/workflows/reusable-ci.yml@phase-2-dev
    with:
      marker: ${{ inputs.marker }}
      python-version: ${{ inputs["python-version"] }}
```

Autofix commits use the configurable prefix (default `chore(autofix):`). Set the
repository variable `AUTOFIX_COMMIT_PREFIX` to change the prefix once and every
workflow picks up the new value. The consolidated Gate workflow consumes the
same reusable entry points, so any new repository can call `reusable-ci.yml`
and `reusable-docker.yml` directly without creating wrappers.

```yaml
name: Agents utilities
on:
  workflow_dispatch:
jobs:
  call:
    uses: stranske/Trend_Model_Project/.github/workflows/reusable-70-agents.yml@phase-2-dev
    with:
      enable_readiness: true
      enable_preflight: true
      enable_watchdog: true
      enable_diagnostic: false
```
Use a tagged ref when versioning downstream consumption.

### Agents Orchestration (Issue #2466)

Issue #2466 finalized the consolidation around the orchestrator. The legacy
consumer/watchdog wrappers are retired; all automation now routes through
`agents-70-orchestrator.yml` and the reusable toolkit it calls. Manual dispatch
steps:

1. Navigate to **Actions → Agents 70 Orchestrator → Run workflow**.
2. Provide the desired inputs (e.g. `enable_bootstrap: true`,
   `bootstrap_issues_label: agent:codex`, `options_json` overrides).
3. Review the `orchestrate` job summary for readiness tables, bootstrap
   planners, watchdog status, and keepalive signals.
4. Rerun as needed; Maint Post CI will echo failing runs in the `ci-failure`
   rollup when Gate is affected.

`reusable-70-agents.yml` remains the single implementation surface for readiness
probes, diagnostics, bootstrap, keepalive, and watchdog jobs. `reuse-agents.yml`
exists for workflow-call reuse so downstream repositories can adopt the same
inputs without duplicating JSON parsing.

---
## 6. Onboarding Checklist (~7 min)

1. Confirm labels `agent:codex`, `agent:copilot`, `autofix`, and `ci-failure`
   exist.
2. Verify repository variables (`OPS_HEALTH_ISSUE`, optional
   `AUTOFIX_OPT_IN_LABEL`) are set.
3. Review Gate and Maint Post CI runs on a recent PR to familiarise yourself
   with the consolidated reporting.
4. Trigger a manual Agents 70 Orchestrator run in dry-run mode (`enable_bootstrap`
   false) to observe readiness output and ensure secrets resolve.
5. Consult `docs/ci/WORKFLOWS.md` for the authoritative workflow roster before
   adding or renaming jobs.

---
## 7. Retired Wrappers

- `agents-consumer.yml`, `agents-41*`, and `agents-42-watchdog.yml` were removed
  during the consolidation. Historical payload examples now live in
  `Old/workflows/` and the repository archive docs.
- `pr-10-ci-python.yml`, `pr-12-docker-smoke.yml`, and the merge-manager flows
  remain archived in `ARCHIVE_WORKFLOWS.md`.

Refer to the archive if you need to resurrect behaviour for forensic analysis;
otherwise, prefer the consolidated orchestrator and reusable workflows.
