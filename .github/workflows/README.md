# Workflow & Agent Automation Quick Start (Issue #2466)

This guide gives maintainers a fast reference for the streamlined CI and agent
automation stack. Pair it with
[docs/WORKFLOW_GUIDE.md](../../docs/WORKFLOW_GUIDE.md) for the canonical
inventory and naming rules.

---
## 1. Architecture Snapshot
Core layers:
- Gate orchestrator (`pr-00-gate.yml`): single required check that fans out to Python 3.11/3.12 CI and the Docker smoke test using the reusable workflows, then enforces that every leg succeeds.
- Autofix lane (`maint-46-post-ci.yml`): workflow_run follower that batches small hygiene fixes, posts Gate summaries, and manages trivial failure remediation using the composite autofix action.
- Agents orchestration & watchdog (`agents-70-orchestrator.yml` + `reusable-70-agents.yml`): label-driven assignment, Codex bootstrap, diagnostics, and watchdog toggles via `enable_watchdog` (default `true`).
- Merge automation (`maint-45-merge-manager.yml`): unified auto-approval and auto-merge decisions for safe agent PRs.
- Governance & Health: `health-40-repo-selfcheck.yml`, `health-41-repo-health.yml`, `health-42-actionlint.yml`, `health-43-ci-signature-guard.yml`, `health-44-gate-branch-protection.yml`, labelers, dependency review, CodeQL.
- Path Labeling: `pr-path-labeler.yml` auto-categorizes PRs.

### 1.1 Current CI Topology (Issue #2439)
The CI stack now routes every pull request through a single Gate workflow that orchestrates the reusable CI and Docker checks:

| Lane | Workflow(s) | Purpose | Required Status Today | Future Plan |
|------|-------------|---------|-----------------------|-------------|
| Gate orchestrator | `pr-00-gate.yml` job `gate` | Coordinates Python (3.11 + 3.12) and Docker smoke runs, fails fast if any leg fails | Required (`Gate / gate`) | Remains the authoritative CI gate |
| Reusable CI | `reusable-10-ci-python.yml` via `pr-00-gate.yml` | Standard Python toolchain (Black, Ruff, mypy, pytest, coverage upload) used by Gate | Called by Gate | Continue to be the single CI entry point |
| Reusable Docker smoke | `reusable-12-ci-docker.yml` via `pr-00-gate.yml` | Deterministic Docker build and smoke probe | Called by Gate | Continue to be the single Docker entry point |
| Autofix lane | `maint-46-post-ci.yml` | Workflow_run follower that posts Gate summaries, commits small hygiene fixes (success runs), and retries trivial CI failures | Not required | Remains optional |

Legacy wrappers (`pr-10-ci-python.yml`, `pr-12-docker-smoke.yml`) have been removed now that branch protection enforces the Gate job directly.

### 1.2 Naming policy & archive status (Issue #1669)
- Active workflows **must** use one of the WFv1 prefixes: `pr-*`, `maint-*`, `agents-*`, or `reusable-*`. Guard tests (`tests/test_workflow_naming.py`) enforce this policy.
- Historical directories `Old/.github/workflows/` and `.github/workflows/archive/` were removed. Reference [ARCHIVE_WORKFLOWS.md](../../ARCHIVE_WORKFLOWS.md) when you need the legacy slugs.
- New workflows should document their purpose in this README and in [WORKFLOW_AUDIT_TEMP.md](../../WORKFLOW_AUDIT_TEMP.md) so future audits inherit a complete inventory.

Flow:
1. PR opened → labelers apply path + agent labels.
2. Labels / branch rules trigger CI, autofix, readiness.
3. Maintainer approval (CODEOWNERS) → `automerge` merges low-risk.
4. Schedules (health, CodeQL) maintain hygiene.

---
## 2. Label Cheat Sheet

| Label | Purpose | Source |
|-------|---------|--------|
| `agent:codex` / `agent:copilot` | Marks automation-owned issues and PRs | Agent labeler |
| `from:codex` / `from:copilot` | Origin marker for automation PRs | Agent labeler |
| `autofix` / `autofix:applied` | Track PR autofix results | Autofix workflow |
| `ci-failure` | Pins the rolling CI dashboard issue | Maint 46 Post CI |
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
|----------|-----------|-------|
| `pr-00-gate.yml` | pull_request, workflow_dispatch | Orchestrates reusable Python 3.11/3.12 CI and Docker smoke tests, then enforces all-success before reporting `gate`.
| `health-41-repo-health.yml` | schedule (weekly), workflow_dispatch | Monday hygiene summary of stale branches and unassigned issues.
| `maint-46-post-ci.yml` | workflow_run (`Gate`) | Consolidated Gate follower for summaries, hygiene autofix, and trivial failure remediation once CI passes.
| `maint-47-check-failure-tracker.yml` | workflow_run (`Gate`) | Opens/resolves CI failure-tracker issues based on run outcomes.
| `health-40-repo-selfcheck.yml` | schedule (daily + weekly), workflow_dispatch | Governance audit that validates labels, PAT availability, and branch protection; maintains a single failure issue when checks fail.
| `health-42-actionlint.yml` | pull_request (workflows), push (`phase-2-dev`), schedule, workflow_dispatch | Workflow schema lint with reviewdog annotations.
| `health-43-ci-signature-guard.yml` | pull_request/push (`phase-2-dev`) | Validates the signed job manifest for `pr-00-gate.yml`.
| `agents-63-chatgpt-issue-sync.yml` | workflow_dispatch | Curated topic lists (e.g. `Issues.txt`) → labeled GitHub issues.
| `maint-45-cosmetic-repair.yml` | workflow_dispatch | Manual pytest + cosmetic fixer that raises guard-gated PRs for tolerated drift.
| `agents-63-codex-issue-bridge.yml` | issues, workflow_dispatch | Prepares Codex-ready branches/PRs when an `agent:codex` label is applied.
| `agents-70-orchestrator.yml` | schedule (*/20), workflow_dispatch | Unified agents toolkit entry point delegating to `reusable-70-agents.yml`.
| `reusable-70-agents.yml` | workflow_call | Composite implementing readiness, bootstrap, diagnostics, and watchdog jobs.
| `reusable-10-ci-python.yml` | workflow_call | Unified CI executor for the Python stack.
| `reusable-12-ci-docker.yml` | workflow_call | Docker smoke reusable consumed by `pr-00-gate.yml`.
| `reusable-92-autofix.yml` | workflow_call | Autofix composite consumed by `maint-46-post-ci.yml`.

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
    uses: stranske/Trend_Model_Project/.github/workflows/reusable-10-ci-python.yml@phase-2-dev
    with:
      marker: ${{ inputs.marker }}
      python-version: ${{ inputs["python-version"] }}
```
Autofix commits use the configurable prefix (default `chore(autofix):`). Set the repository variable
`AUTOFIX_COMMIT_PREFIX` to change the prefix once and every workflow picks up the new value. The
consolidated Gate workflow consumes the same reusable entry points, so any new repository can call
`reusable-10-ci-python.yml` and `reusable-12-ci-docker.yml` directly without needing an intermediate wrapper.

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
Use a tagged ref when versioned.

### Agents Orchestration (Issue #2377)
Issue #2377 rebuilt the agents automation stack to stay under the GitHub
`workflow_dispatch` input cap while restoring legacy consumer behaviour.
Two entry points now exist:

- `agents-62-consumer.yml` – Manual dispatch wrapper that accepts a single
  `params_json` string, parses it, and forwards normalized values to
  `reusable-71-agents-dispatch.yml`. The workflow declares
  `concurrency: agents-62-consumer` and introduces job-level
  `timeout-minutes` so overlapping runs are cancelled and stalled executions
  end automatically. Set `enable_bootstrap` to `true` in the JSON payload to
  opt into Codex bootstraps (preflight stays disabled unless explicitly
  enabled).
- `agents-70-orchestrator.yml` – Unified scheduled/dispatch orchestrator for
  readiness probes, diagnostics, bootstrap, watchdog, and keepalive flows. It
  passes discrete inputs directly to `reusable-70-agents.yml` and derives
  Codex bootstrap toggles/labels from the `options_json` payload so the
  dispatch form stays under the 10-input limit.

`reusable-71-agents-dispatch.yml` bridges the consumer JSON payload into the reusable toolkit
without re-exposing more than 10 dispatch inputs. Both entry points ultimately
invoke `reusable-70-agents.yml`, which emits Markdown readiness summaries,
`issue_numbers_json`, and `first_issue` outputs for Codex bootstraps and keeps
the watchdog probe enabled whenever `enable_watchdog` resolves to `true`.

Manual dispatch for the consumer now uses a single JSON textarea. A ready to
paste payload:

```json
{
  "enable_readiness": true,
  "readiness_agents": "copilot,codex",
  "custom_logins": "",
  "require_all": false,
  "enable_preflight": false,
  "codex_user": "",
  "codex_command_phrase": "",
  "enable_verify_issue": false,
  "verify_issue_number": "",
  "enable_watchdog": true,
  "enable_bootstrap": false,
  "bootstrap_issues_label": "agent:codex",
  "draft_pr": false,
  "options_json": "{\"enable_keepalive\":false,\"keepalive\":{\"enabled\":false}}"
}
```

Omit any keys to fall back to defaults. `enable_bootstrap: true` unlocks Codex
PR bootstraps; leave it `false` for the minimal readiness + watchdog run.
`options_json` remains available for advanced keepalive tuning (dry run,
alternate labels, idle thresholds, etc.). Set `enable_bootstrap: true`
alongside an optional `bootstrap_issues_label` (or `bootstrap: { "label":
"..." }`) in `options_json` to turn on Codex bootstraps for orchestrator
dispatches. Leave `enable_keepalive` (or `keepalive.enabled`) set to `false`
to keep the sweep disabled for scheduled consumer runs.

The guard test `tests/test_workflow_agents_consolidation.py` enforces the
reduced input surface and ensures the consumer continues to call the bridge
workflow. Update the README whenever adding new JSON keys so operators have an
accurate dispatch reference.

### Merge Manager (Issue #1415)
Unified approval + auto-merge policy lives in `maint-45-merge-manager.yml`, replacing the legacy pair `autoapprove.yml` and
`enable-automerge.yml` (retired; historical details tracked in `ARCHIVE_WORKFLOWS.md`). Guard test: `tests/test_workflow_merge_manager.py`.

Design invariants:
1. Single rationale comment per PR identified by marker `<!-- merge-manager-rationale -->`.
2. Combined safety evaluation (allowlist patterns, size cap, quiet period, active workflow absence) before any approval.
3. Conditional approval (GitHub Review API) + optional auto-merge enablement via `peter-evans/enable-pull-request-automerge@v3`.
4. Loop guard: declines if last commit already bears the autofix prefix (`COMMIT_PREFIX`, default `chore(autofix):`).
5. Idempotent: re-runs update / replace the existing rationale comment instead of spamming.

Acceptance Criteria (Issue #1415) satisfied by: archival of legacy workflows, presence of guard test, README documentation, and operational unified workflow.

---
## 6. Onboarding Checklist (~7m)
1. Create labels `automerge`, `risk:low`, `agent:codex`, `agent:copilot`, `codex-ready`.
2. Add area labels.
3. Add `SERVICE_BOT_PAT` or set `CODEX_ALLOW_FALLBACK=true` (temporary).
4. Ensure Actions write permission.
5. Add CI / Autofix / Agents consumers.
6. Open dummy PR → verify labels.
7. Dispatch readiness.

---
## 7. Troubleshooting
| Symptom | Cause | Ref |
|---------|-------|-----|
| No labels | Labeler/perms missing | `pr-02-label-agent-prs.yml` |
| Bootstrap blocked | PAT missing & fallback off | troubleshooting doc |
| Autofix skipped | Title match / opt-in absent | Autofix README |
| No dependency review | Fork PR / disabled | `pr-31-dependency-review.yml` |
| No CodeQL alerts | First run indexing | `pr-30-codeql.yml` |

### 7.1 Autofix Loop Guard (Issue #1347)
Loop prevention layers:
1. The consolidated workflow only reacts to completed CI runs (no direct `push` trigger).
2. Guard logic only fires when the workflow actor is `github-actions` (or `github-actions[bot]`) **and** the latest commit subject begins with the standardized prefix `chore(autofix):`.
3. Scheduled cleanup (`maint-31-autofix-residual-cleanup.yml`) and reusable autofix consumers adopt the same prefix + actor guard, so automation commits short-circuit immediately instead of chaining runs.
4. The CI style job runs independently and does not trigger autofix.

Result: Each human push generates at most one autofix patch sequence; autofix commits do not recursively spawn new runs.

---
## 7.2 Codex Kickoff Flow (Issue #1351)
End‑to‑end lifecycle for automation bootstrapped contributions:
1. Maintainer opens Issue with label `codex-ready` (and optional spec details).
2. Labeling with `agent:codex` triggers `agents-41-assign.yml`, which creates a bootstrap branch/PR, assigns Codex, and posts the kickoff command.
3. `agents-42-watchdog.yml` (dispatched by the assigner) waits ~7 minutes for the cross-referenced PR and posts a success or timeout diagnostic comment.
4. When automation pushes commits, path labelers, CI, and autofix re-evaluate.
Troubleshooting: If branch/PR not created, verify the label `codex-ready`, confirm `agents-41-assign.yml` completed successfully with write permissions, and ensure no conflicting bootstrap branch already exists.

---
## 7.3 Coverage Soft Gate (Issues #1351, #1352)
Purpose: Provide early visibility of coverage / hotspot data without failing PRs.


Low Coverage Spotlight (follow-up Issue #1386):
- A secondary table "Low Coverage (<X%)" appears when any parsed file has coverage below the configured threshold (default 50%).
- Customize the threshold with the `low-coverage-threshold` workflow input when calling `reusable-10-ci-python.yml`.
- Table is separately truncated to the hotspot limit (15) with a truncation notice if more remain.
Implemented follow-ups (Issue #1352):
- Normalized artifact naming: `coverage-<python-version>` (e.g. `coverage-3.11`).
- Consistent file set per matrix job: `coverage.xml`, `coverage.json`, `htmlcov/**`, `pytest-junit.xml`, `pytest-report.xml`.
- Retention window input `coverage-artifact-retention-days` has a default value of 10.
  This default is chosen to fall within the recommended 7–14 day observation horizon, allowing reviewers to compare multiple consecutive runs without long-term storage bloat.
  Adjust as needed; it is suggested to keep the retention window within 7–14 days unless you are auditing longer-term trends.
- Single canonical coverage tracking issue auto-updated with summary + hotspots + job log links.
- Run Summary includes a single "Soft Coverage Gate" section (job log table de-duplicated into universal logs job).
- Trend artifacts shipped: `coverage-trend.json` (single run) and cumulative `coverage-trend-history.ndjson` (history) for longitudinal analysis.

Activation (consumer of `reusable-10-ci-python.yml`):
```yaml
with:
  enable-soft-gate: 'true'
```
Outputs:
- Run Summary section: "Soft Coverage Gate" with average coverage (across matrix), worst job, and top 15 lowest-covered files (hotspots).
- Artifacts (per Python version): `coverage-<ver>` bundle (xml/json/htmlcov + JUnit variants) retained N days (default 10).
- Aggregated artifacts: `coverage-trend` (JSON for this run), `coverage-trend-history` (NDJSON accumulating all runs).
- Canonical coverage Issue comment (create-or-update) containing run link, summary, hotspots, and job log links (deduped from summary table).

### Agents Orchestration (Issue #2466)

---
## 7.4 Self-Test Reusable CI (Issue #1660)

- **Trigger scope:** Manual dispatch plus a weekly cron (`Mondays @ 06:00 UTC`). This keeps reusable pipeline coverage fresh
  without consuming PR minutes. Dispatch from the Actions tab under **Self-Test Reusable CI** or via CLI when validating
  changes to `.github/workflows/reusable-10-ci-python.yml` or its helper scripts. The legacy PR-comment notifier was removed because the
  workflow no longer runs on pull_request events. After shipping a change, monitor the next two scheduled runs and confirm
  their success in the self-test health issue before considering the work complete.
- **Latest remediation:** The October 2025 failure stemmed from `typing-inspection` drifting from `0.4.1` to `0.4.2`, causing
  `tests/test_lockfile_consistency.py` to fail during the reusable matrix runs. Refresh `requirements.lock` with
  `uv pip compile --upgrade pyproject.toml -o requirements.lock` before re-running the workflow. The matrix now completes when
  invoked manually or on schedule.
- **Diagnostics:** Each run uploads a `selftest-report` artifact summarising scenario coverage and any unexpected or missing
  artifacts. Use it alongside the job logs to validate new reusable features before promoting changes.
- **Failure triage workflow:** When a nightly run fails, open the run in the Actions tab and download diagnostics with the
  GitHub CLI:

  ```bash
  gh run download <run-id> --dir selftest-artifacts
  gh run view <run-id> --log
  ```

  Inspect `selftest-artifacts/selftest-report/selftest-report.json` for mismatched artifacts and reproduce dependency drift
  issues locally or to validate lockfile drift fixes with `pytest tests/test_lockfile_consistency.py -k "up_to_date" -q`.

- **CLI helpers:**

  ```bash
  # Manual dispatch (requires `gh auth login` with workflow scope)
  gh workflow run "Self-Test Reusable CI"

  # Monitor most recent nightly outcomes
  gh run list --workflow "Self-Test Reusable CI" --limit 2 --json conclusion,headBranch,runNumber,startedAt
  ```

  Use the `gh run list` output to confirm two consecutive nightly runs have concluded with `success` before closing out
  Issue #1660 follow-up tasks.
## 7.5 Universal Logs Summary (Issue #1351)
Source: `logs_summary` job inside `reusable-10-ci-python.yml` enumerates all jobs via the Actions API and writes a Markdown table to the run summary. Columns include Job, Status (emoji), Duration, and Log link.

How to access logs:
1. Open the PR → Checks tab → select the CI run.
2. Scroll to the Run Summary table; click the log link for any job.
3. Fallback: Use the GitHub UI Jobs list if the summary table is missing.

If missing:
- Confirm the `logs_summary` job executed (it is unconditional). If skipped, check for GitHub API rate limits in its step logs.

---
## 7.6 Gate-Only Protection (Issue #2439)
Branch protection now requires the `Gate / gate` job directly. The historical wrappers have been removed and all automation listens to the Gate workflow via `workflow_run` triggers. No further action is required beyond keeping the reusable workflows healthy.

---
## 7.7 Quick Reference – Coverage & Logs
| Concern | Job / File | How to Enable | Artifact / Output | Fails Build? |
|---------|------------|---------------|-------------------|--------------|
| Coverage soft gate | Job: `coverage_soft_gate` in `reusable-10-ci-python.yml` | `enable-soft-gate: 'true'` | Run summary section, coverage artifacts | No |
| Universal logs table | Job: `logs_summary` | Always on | Run summary Markdown table | No |
| Gate aggregation | Job: `gate` in `pr-00-gate.yml` | Always on | Single pass/fail gate | Yes (required) |

Note: The gate job will become the only required status after successful observation window.


---
## 8. Extensibility
- Add quarantine job via new inputs.
- Tune dependency severity gating.
- Tag releases for stable reuse.

1. Navigate to **Actions → Agents 70 Orchestrator → Run workflow**.
2. Provide the desired inputs (e.g. `enable_bootstrap: true`,
   `bootstrap_issues_label: agent:codex`, `options_json` overrides).
3. Review the `orchestrate` job summary for readiness tables, bootstrap
   planners, watchdog status, and keepalive signals.
4. Rerun as needed; Maint 46 Post CI will echo failing runs in the `ci-failure`
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
3. Review Gate and Maint 46 Post CI runs on a recent PR to familiarise yourself
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
