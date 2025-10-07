# Workflow & Agent Automation Quick Start (Issue #1204)

This guide enables a new maintainer to operate the CI + agent automation stack in under 10 minutes. For a reference catalogue of naming rules, workflow buckets, and agent label guidance see [docs/WORKFLOW_GUIDE.md](../../docs/WORKFLOW_GUIDE.md).

---
## 1. Architecture Snapshot
Core layers:
- CI python (`pr-10-ci-python.yml`): single job that installs the pinned toolchain, runs Black, Ruff, mypy, pytest with coverage, uploads diagnostics, and publishes a coverage summary.
- Autofix lane (`maint-32-autofix.yml`): workflow_run follower that batches small hygiene fixes and trivial failure remediation using the composite autofix action.
- Agent routing & watchdog (`agents-41-assign.yml` + `agents-42-watchdog.yml`): label-driven assignment, Codex bootstrap, diagnostics.
- Merge automation (`maint-45-merge-manager.yml`): unified auto-approval and auto-merge decisions for safe agent PRs.
- Governance & Health: `maint-34-quarantine-ttl.yml`, `maint-35-repo-health-self-check.yml`, `maint-36-actionlint.yml`, labelers, dependency review, CodeQL.
- Path Labeling: `pr-path-labeler.yml` auto-categorizes PRs.

### 1.1 Current CI Topology (Issue #2195)
The CI stack now keeps the Python checks inside a single job while the supporting lanes remain independent:

| Lane | Workflow(s) | Purpose | Required Status Today | Future Plan |
|------|-------------|---------|-----------------------|-------------|
| Core python | `pr-10-ci-python.yml` job `ci / python` | Black, Ruff, mypy, pytest, coverage summary & threshold | Required (`CI python`) | Remains the authoritative CI gate |
| Autofix lane | `maint-32-autofix.yml` | Workflow_run follower that commits small hygiene fixes (success runs) and retries trivial CI failures | Not required | Remains optional |
| Agent assignment | `agents-41-assign.yml` | Maps labels → assignees, creates Codex bootstrap PRs | Not required | Harden diagnostics |
| Agent watchdog | `agents-42-watchdog.yml` | Confirms Codex PR cross-reference or posts timeout | Not required | Tune timeout post burn-in |

The historical multi-job topology (style/type gate + matrix tests feeding a `gate / all-required-green` aggregator) has been replaced by the single `ci / python` job. Documentation and automation that referenced the old job names were updated in this PR to prevent drift.

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
| `agent:codex` / `agent:copilot` | Automation origin PRs | Agent labeler |
| `from:codex` / `from:copilot` | Origin marker | Agent labeler |
| `risk:low` | Low-friction auto-merge | Issue form / labeler |
| `automerge` | Eligible for merge automation | Issue form / maintainer |
| `codex-ready` | Allows bootstrap run | Issue template |
| `type:bug` / `type:feature` | Taxonomy | Templates |
| Area labels | Scope classification | Path labeler |

---
## 3. Required Secrets & Variables
| Name | Type | Req | Purpose | Notes |
|------|------|-----|---------|-------|
| `SERVICE_BOT_PAT` | Secret | Rec | Cross-fork ops identity | `repo` scope |
| `CODEX_ALLOW_FALLBACK` | Var | Opt | Allow fallback token use | Temporary only |
| `AUTOMERGE_LABEL` | Var | Opt | Customize `automerge` label | Must exist |
| `RISK_LABEL` | Var | Opt | Customize risk label | Default `risk:low` |
| `AGENT_LABEL` / `AGENT_LABEL_ALT` | Var | Opt | Agent classification synonyms | Cosmetic |
| `AUTOFIX_OPT_IN_LABEL` | Var | Opt | Gate autofix | Falls back internally |
| `OPS_HEALTH_ISSUE` | Var | Req | Issue number for nightly health updates | Validate the value exists in every environment; repo-health skips issue updates when unset. |

All others use default `GITHUB_TOKEN`.

---
## 4. Trigger Matrix
| Workflow | Trigger(s) | Notes |
|----------|-----------|-------|
| `pr-10-ci-python.yml` | pull_request, push, workflow_call | Unified Python checks (Black, Ruff, mypy, pytest, coverage summary).
| `pr-12-docker-smoke.yml` | pull_request, push | Deterministic Docker build followed by a health endpoint smoke test.
| `maint-02-repo-health.yml` | schedule (weekly), workflow_dispatch | Monday hygiene summary of stale branches and unassigned issues.
| `maint-30-post-ci-summary.yml` | workflow_run (`pr-10`, `pr-12`) | Publishes consolidated CI status for active PRs.
| `maint-32-autofix.yml` | workflow_run (`pr-10`, `pr-12`) | Hygiene autofix plus trivial failure remediation once CI passes.
| `maint-33-check-failure-tracker.yml` | workflow_run (`pr-10`, `pr-12`) | Opens/resolves CI failure-tracker issues based on run outcomes.
| `maint-35-repo-health-self-check.yml` | schedule (daily + weekly), workflow_dispatch | Governance audit that validates labels, PAT availability, and branch protection; maintains a single failure issue when checks fail.
| `maint-36-actionlint.yml` | pull_request (workflows), push (`phase-2-dev`), schedule, workflow_dispatch | Workflow schema lint with reviewdog annotations.
| `maint-40-ci-signature-guard.yml` | pull_request/push (`phase-2-dev`) | Validates the signed job manifest for `pr-10-ci-python.yml`.
| `maint-41-chatgpt-issue-sync.yml` | workflow_dispatch | Curated topic lists (e.g. `Issues.txt`) → labeled GitHub issues.
| `agents-43-codex-issue-bridge.yml` | issues, workflow_dispatch | Prepares Codex-ready branches/PRs when an `agent:codex` label is applied.
| `agents-70-orchestrator.yml` | schedule (*/20), workflow_dispatch | Unified agents toolkit entry point delegating to `reusable-70-agents.yml`.
| `reusable-70-agents.yml` | workflow_call | Composite implementing readiness, bootstrap, diagnostics, and watchdog jobs.
| `reusable-90-ci-python.yml` | workflow_call | Unified CI executor for the Python stack.
| `reusable-92-autofix.yml` | workflow_call | Autofix composite consumed by `maint-32-autofix.yml`.
| `reusable-94-legacy-ci-python.yml` | workflow_call | Compatibility shim for repositories that still need the legacy matrix layout.

---
## 5. Adopt Reusable Workflows
CI consumer:
```yaml
name: CI
on:
  pull_request:
  push:
    branches: [ main ]
jobs:
  call:
    uses: stranske/Trend_Model_Project/.github/workflows/pr-10-ci-python.yml@phase-2-dev
```
Autofix commits use the configurable prefix (default `chore(autofix):`). Set the repository variable
`AUTOFIX_COMMIT_PREFIX` to change the prefix once and every workflow picks up the new value. The
consolidated workflow guards against loops by detecting automation actors + existing prefix and only
running after the CI workflow completes. Scheduled cleanup and reusable autofix helpers consume the
same prefix so the guard behaviour is identical no matter which workflow authored the last automation
commit.

```yaml
name: Agents utilities
on:
  workflow_dispatch:
jobs:
  call:
    uses: stranske/Trend_Model_Project/.github/workflows/reusable-90-agents.yml@phase-2-dev
    with:
      enable_readiness: true
      enable_preflight: true
      enable_diagnostic: false
```
Use a tagged ref when versioned.

### Agents Orchestration (Issue #2377)
Issue #2377 rebuilt the agents automation stack to stay under the GitHub
`workflow_dispatch` input cap while restoring legacy consumer behaviour.
Two entry points now exist:

- `agents-consumer.yml` – Hourly cron + manual dispatch wrapper that accepts a
  single `params_json` string, parses it, and forwards normalized values to
  `reuse-agents.yml`. Scheduled runs only execute readiness + watchdog probes;
  set `enable_bootstrap` to `true` in the JSON payload to opt into Codex
  bootstraps (preflight stays disabled unless explicitly enabled).
- `agents-70-orchestrator.yml` – Unified scheduled/dispatch orchestrator for
  readiness probes, diagnostics, bootstrap, watchdog, and keepalive flows. It
  passes discrete inputs directly to `reusable-70-agents.yml`.

`reuse-agents.yml` bridges the consumer JSON payload into the reusable toolkit
without re-exposing more than 10 dispatch inputs. Both entry points ultimately
invoke `reusable-70-agents.yml`, which emits Markdown readiness summaries,
`issue_numbers_json`, and `first_issue` outputs for Codex bootstraps.

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
alternate labels, idle thresholds, etc.).

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
- Customize the threshold with the `low-coverage-threshold` workflow input when calling `reusable-90-ci-python.yml`.
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

Activation (consumer of `reusable-90-ci-python.yml`):
```yaml
with:
  enable-soft-gate: 'true'
```
Outputs:
- Run Summary section: "Soft Coverage Gate" with average coverage (across matrix), worst job, and top 15 lowest-covered files (hotspots).
- Artifacts (per Python version): `coverage-<ver>` bundle (xml/json/htmlcov + JUnit variants) retained N days (default 10).
- Aggregated artifacts: `coverage-trend` (JSON for this run), `coverage-trend-history` (NDJSON accumulating all runs).
- Canonical coverage Issue comment (create-or-update) containing run link, summary, hotspots, and job log links (deduped from summary table).

Behavior: Non‑blocking (always succeeds). Parsing failures degrade gracefully (warning + skip) to avoid blocking unrelated PR progress.
Hotspots: Sorted ascending by percent covered (lowest coverage first) limited to 15 entries for scannability.
Retention Guidance: Use 7–14 days. Shorter (<7 days) risks losing comparison context for slower review cycles; longer (>14 days) increases storage without materially improving triage.

---
## 7.4 Self-Test Reusable CI (Issue #1660)

- **Trigger scope:** Manual dispatch plus a nightly cron (`02:30 UTC`). This keeps reusable pipeline coverage fresh without
  consuming PR minutes. Dispatch from the Actions tab under **Self-Test Reusable CI** or via CLI when validating changes to
  `.github/workflows/reusable-90-ci-python.yml` or its helper scripts. The legacy PR-comment notifier was removed because the
  workflow no longer runs on pull_request events. After shipping a change, monitor the next two nightly runs and confirm
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
Source: `logs_summary` job inside `reusable-90-ci-python.yml` enumerates all jobs via the Actions API and writes a Markdown table to the run summary. Columns include Job, Status (emoji), Duration, and Log link.

How to access logs:
1. Open the PR → Checks tab → select the CI run.
2. Scroll to the Run Summary table; click the log link for any job.
3. Fallback: Use the GitHub UI Jobs list if the summary table is missing.

If missing:
- Confirm the `logs_summary` job executed (it is unconditional). If skipped, check for GitHub API rate limits in its step logs.

---
## 7.6 Temporary CI Wrapper & Migration Plan (Issue #1351)
`pr-10-ci-python.yml` wraps the reusable CI to maintain the historical required check label "CI".

Migration steps to retire wrapper:
1. Add `gate / all-required-green` job as a required status alongside "CI" in branch protection.
2. Observe stability for N (suggested: 7–14) days (no unexplained gate misses).
3. Remove "CI" from required list, leaving the gate job.
4. Delete `pr-10-ci-python.yml` in a dedicated PR referencing Issue #1351 (or follow-up) and update this README (remove this section).
5. Re-run a test PR to ensure branch protection enforces the gate job.

Rationale: Allows a staged transition without breaking existing protections.

---
## 7.7 Quick Reference – Coverage & Logs
| Concern | Job / File | How to Enable | Artifact / Output | Fails Build? |
|---------|------------|---------------|-------------------|--------------|
| Coverage soft gate | Job: `coverage_soft_gate` in `reusable-90-ci-python.yml` | `enable-soft-gate: 'true'` | Run summary section, coverage artifacts | No |
| Universal logs table | Job: `logs_summary` | Always on | Run summary Markdown table | No |
| Gate aggregation | Job: `gate / all-required-green` | Always on | Single pass/fail gate | Yes (if made required) |
| Legacy wrapper | `pr-10-ci-python.yml` | N/A | Preserves required check name | N/A |

Note: The gate job will become the only required status after successful observation window.


---
## 8. Extensibility
- Add quarantine job via new inputs.
- Tune dependency severity gating.
- Tag releases for stable reuse.

---
## 9. Deep-Dive Docs
| Topic | Doc |
|-------|-----|
| Reusable design | `docs/ci_reuse.md` |
| Consolidation | `docs/ci_reuse_consolidation_plan.md` |
| Agent modes | `docs/agent-automation.md` |
| Bootstrap verify | `docs/codex_bootstrap_verification.md` |
| Troubleshooting | `docs/agent_codex_troubleshooting.md` |
| Bot facts | `docs/ops/codex-bootstrap-facts.md` |

---
## 10. Change Process
Update this README + workflows in PR; note semantic changes inline as design notes.

---
## 11. Stale PR TTL (Issue #1205)
`stale-prs.yml` (daily 02:23 UTC + manual)

Defaults:
- Warn after 14d inactivity (`stale` label).
- Close after 21d inactivity.
- Exempt: `pinned`, `work-in-progress`, `security`, `blocked`.
Activity clears `stale`.

Tips: long draft → add `work-in-progress`; external wait → `blocked`.
Tune via `days-before-pr-stale` / `days-before-pr-close`.

Future (planned): telemetry summary, org-level TTL var.

_Last updated: 2025-09-19 (Issue #1205)_

---
## 13. Future Enhancements (Advisory)
Planned / optional improvements under consideration:
| Enhancement | Status | Notes |
|-------------|--------|-------|
| Coverage trend artifact (JSON) | Implemented | `coverage-trend` provides run-level stats (Issue #1352) |
| Coverage trend history (NDJSON) | Implemented | `coverage-trend-history` accumulates per-run records |
| CI style job (ruff+black+mypy) | Implemented | Replaces legacy lint-verification (flake8/black) |
| Centralized autofix commit prefix | Implemented | Configurable (default `chore(autofix):`) |
| Failing test count in logs summary | Implemented | Universal logs job appends count inline |

TODO (wrapper removal): After branch protection flips to require the gate job, remove `pr-10-ci-python.yml` (see 7.6) and delete this TODO line.

Adopt individually; update sections 7.3 / 7.5 when shipped.

---
_Addendum (Issues #1351, #1352): CI topology, kickoff flow, soft gate, logs summary, coverage artifact normalization, trend history, and migration plan documented. Wrapper removal pending future protection flip._

---
## 12. Agent Readiness Enhancements (Issue #1220)
Richer readiness probing.

New Inputs:
- `readiness_custom_logins`: comma-separated bot usernames.
- `require_all`: fail if any requested builtin or custom login missing when true.

Existing:
- `readiness_agents`: builtin keys (`copilot,codex`).

Outputs:
- Markdown table + JSON block (summary).
- Columns: Agent | Kind | Requested | Assignable | Resolved Login.

Failure Semantics:
- `require_all=false` → always succeed (missing show ❌).
- `require_all=true` → fail on any missing.

Example:
```yaml
with:
  enable_readiness: 'true'
  readiness_agents: 'copilot,codex'
  readiness_custom_logins: 'my-internal-bot'
  require_all: 'true'
```

Rationale: Portability across repos + deterministic artifacts.

_Last updated: 2025-09-19 (Issue #1220)_
