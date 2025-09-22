# Workflow & Agent Automation Quick Start (Issue #1204)

This guide enables a new maintainer to operate the CI + agent automation stack in under 10 minutes.

---
## 1. Architecture Snapshot
Core layers:
- Reusable CI (`reuse-ci-python.yml`): tests, coverage, aggregated gate.
- Autofix lane (`autofix.yml`): workflow_run follower that batches small hygiene fixes and trivial failure remediation using the composite autofix action.
- Style Gate (`style-gate.yml`): authoritative style verification (black --check + ruff new-issue fail) running on PR & main branch pushes.
- Agent routing & watchdog (`assign-to-agents.yml` + `agent-watchdog.yml`): label-driven assignment, Codex bootstrap, diagnostics.
- Merge automation (`merge-manager.yml`): unified auto-approval and auto-merge decisions for safe agent PRs.
- Governance & Health: `repo-health-self-check.yml`, labelers, dependency review, CodeQL.
- Path Labeling: `pr-path-labeler.yml` auto-categorizes PRs.

### 1.1 Current CI Topology (Issue #1351)
The CI stack now runs in distinct lanes so each concern can evolve independently:

| Lane | Workflow(s) | Purpose | Required Status Today | Future Plan |
|------|-------------|---------|-----------------------|-------------|
| Core test/coverage | `reusable-ci-python.yml` (consumed by `ci.yml`) | Matrix tests, coverage report | Wrapper job "CI" (legacy) + gate job | Make gate job the only required check once stable |
| Gate aggregation | `reusable-ci-python.yml` job: `gate / all-required-green` | Ensures upstream jobs passed (single source of truth) | Secondary | Will replace wrapper after burn‑in |
| Coverage soft gate | `coverage_soft_gate` job (opt‑in) | Posts coverage & hotspots (non-blocking) | Advisory | Remains advisory |
| Universal logs | `logs_summary` job | Per‑job log table in summary | Not required | Always-on helper |
| Autofix lane | `autofix.yml` | Workflow_run follower that commits small hygiene fixes (success runs) and retries trivial CI failures | Not required | Remains optional |
| Style verification | `style-gate.yml` | Enforce black formatting + ruff cleanliness (fail on new issues) | Candidate required | Become required once stable |
| Agent assignment | `assign-to-agents.yml` | Maps labels → assignees, creates Codex bootstrap PRs | Not required | Harden diagnostics |
| Agent watchdog | `agent-watchdog.yml` | Confirms Codex PR cross-reference or posts timeout | Not required | Tune timeout post burn-in |

Temporary state: `ci.yml` exists solely to preserve the historic required check name ("CI") while maintainers transition branch protection to the gate job. Once maintainers flip protection, delete `ci.yml` and mark the gate job required.

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

All others use default `GITHUB_TOKEN`.

---
## 4. Trigger Matrix
| Workflow | Trigger(s) | Notes |
|----------|-----------|-------|
| `reuse-ci-python.yml` | PR, push | Coverage & matrix |
| `autofix.yml` | workflow_run (`CI`) | Hygiene autofix + trivial failure remediation |
| `style-gate.yml` | PR, push (main branches) | Style enforcement |
| `assign-to-agents.yml` | issue/PR labels, dispatch | Agent assignment + Codex bootstrap |
| `agent-watchdog.yml` | workflow dispatch | Codex PR presence diagnostic |
| `merge-manager.yml` | PR target, workflow_run | Auto-approve + enable auto-merge when gates are satisfied |
| `repo-health-self-check.yml` | schedule, manual | Governance audit |
| `pr-path-labeler.yml` | PR events | Path labels |
| `label-agent-prs.yml` | PR target | Origin + risk labels |
| `codeql.yml` | push, PR, schedule | Code scanning |
| `dependency-review.yml` | PR | Dependency diff gate |

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
    uses: stranske/Trend_Model_Project/.github/workflows/reuse-ci-python.yml@phase-2-dev
    with:
      python_matrix: '"3.11"'
      cov_min: 70
```
Autofix commits use the configurable prefix (default `chore(autofix):`). Set the repository variable
`AUTOFIX_COMMIT_PREFIX` to change the prefix once and every workflow picks up the new value. The
consolidated workflow guards against loops by detecting automation actors + existing prefix and only
running after the CI workflow completes. Scheduled cleanup and reusable autofix helpers consume the
same prefix so the guard behaviour is identical no matter which workflow authored the last automation
commit.

```yaml
name: Agents
on:
  workflow_dispatch:
  pull_request:
    types: [opened, synchronize]
jobs:
  call:
    uses: stranske/Trend_Model_Project/.github/workflows/reuse-agents.yml@phase-2-dev
    with:
      enable_readiness: true
      enable_preflight: true
      enable_diagnostic: false
```
Use a tagged ref when versioned.

### Consolidation (Issue #1419)
Active agent automation is intentionally reduced to two workflows:

- `assign-to-agents.yml` – Assigns the appropriate agent on label, boots Codex issue branches, posts trigger commands, and dispatches the watchdog.
- `agent-watchdog.yml` – Polls the issue timeline for a cross‑referenced PR and reports success or a precise timeout.

Legacy orchestrators (`agents-consumer.yml`, `reuse-agents.yml`) are archived under `Old/.github/workflows/` and guarded by
`tests/test_workflow_agents_consolidation.py` to prevent silent reintroduction. Any new agent helper must either extend the
existing assigner or document a justification for a third workflow in this README.

### Merge Manager (Issue #1415)
Unified approval + auto-merge policy lives in `merge-manager.yml`, replacing the legacy pair `autoapprove.yml` and
`enable-automerge.yml` (now archived under `Old/.github/workflows/`). Guard test: `tests/test_workflow_merge_manager.py`.

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
| No labels | Labeler/perms missing | `label-agent-prs.yml` |
| Bootstrap blocked | PAT missing & fallback off | troubleshooting doc |
| Autofix skipped | Title match / opt-in absent | Autofix README |
| No dependency review | Fork PR / disabled | `dependency-review.yml` |
| No CodeQL alerts | First run indexing | `codeql.yml` |

### 7.1 Autofix Loop Guard (Issue #1347)
Loop prevention layers:
1. The consolidated workflow only reacts to completed CI runs (no direct `push` trigger).
2. Guard logic only fires when the workflow actor is `github-actions` (or `github-actions[bot]`) **and** the latest commit subject begins with the standardized prefix `chore(autofix):`.
3. Scheduled cleanup (`autofix-residual-cleanup.yml`) and reusable autofix consumers adopt the same prefix + actor guard, so automation commits short-circuit immediately instead of chaining runs.
4. Style Gate runs independently and does not trigger autofix.

Result: Each human push generates at most one autofix patch sequence; autofix commits do not recursively spawn new runs.

---
## 7.2 Codex Kickoff Flow (Issue #1351)
End‑to‑end lifecycle for automation bootstrapped contributions:
1. Maintainer opens Issue with label `codex-ready` (and optional spec details).
2. Labeling with `agent:codex` triggers `assign-to-agents.yml`, which creates a bootstrap branch/PR, assigns Codex, and posts the kickoff command.
3. `agent-watchdog.yml` (dispatched by the assigner) waits ~7 minutes for the cross-referenced PR and posts a success or timeout diagnostic comment.
4. When automation pushes commits, path labelers, CI, and autofix re-evaluate.
Troubleshooting: If branch/PR not created, verify the label `codex-ready`, confirm `assign-to-agents.yml` completed successfully with write permissions, and ensure no conflicting bootstrap branch already exists.

---
## 7.3 Coverage Soft Gate (Issues #1351, #1352)
Purpose: Provide early visibility of coverage / hotspot data without failing PRs.


Low Coverage Spotlight (follow-up Issue #1386):
- A secondary table "Low Coverage (<X%)" appears when any parsed file has coverage below the configured threshold (default 50%).
- Customize the threshold with the `low-coverage-threshold` workflow input when calling `reusable-ci-python.yml`.
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

Activation (consumer of `reusable-ci-python.yml`):
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
## 7.4 Universal Logs Summary (Issue #1351)
Source: `logs_summary` job inside `reusable-ci-python.yml` enumerates all jobs via the Actions API and writes a Markdown table to the run summary. Columns include Job, Status (emoji), Duration, and Log link.

How to access logs:
1. Open the PR → Checks tab → select the CI run.
2. Scroll to the Run Summary table; click the log link for any job.
3. Fallback: Use the GitHub UI Jobs list if the summary table is missing.

If missing:
- Confirm the `logs_summary` job executed (it is unconditional). If skipped, check for GitHub API rate limits in its step logs.

---
## 7.5 Temporary CI Wrapper & Migration Plan (Issue #1351)
`ci.yml` wraps the reusable CI to maintain the historical required check label "CI".

Migration steps to retire wrapper:
1. Add `gate / all-required-green` job as a required status alongside "CI" in branch protection.
2. Observe stability for N (suggested: 7–14) days (no unexplained gate misses).
3. Remove "CI" from required list, leaving the gate job.
4. Delete `ci.yml` in a dedicated PR referencing Issue #1351 (or follow-up) and update this README (remove this section).
5. Re-run a test PR to ensure branch protection enforces the gate job.

Rationale: Allows a staged transition without breaking existing protections.

---
## 7.6 Quick Reference – Coverage & Logs
| Concern | Job / File | How to Enable | Artifact / Output | Fails Build? |
|---------|------------|---------------|-------------------|--------------|
| Coverage soft gate | Job: `coverage_soft_gate` in `reusable-ci-python.yml` | `enable-soft-gate: 'true'` | Run summary section, coverage artifacts | No |
| Universal logs table | Job: `logs_summary` | Always on | Run summary Markdown table | No |
| Gate aggregation | Job: `gate / all-required-green` | Always on | Single pass/fail gate | Yes (if made required) |
| Legacy wrapper | `ci.yml` | N/A | Preserves required check name | N/A |

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
| Style Gate (ruff+black) | Implemented | Replaces legacy lint-verification (flake8/black) |
| Centralized autofix commit prefix | Implemented | Configurable (default `chore(autofix):`) |
| Failing test count in logs summary | Implemented | Universal logs job appends count inline |

TODO (wrapper removal): After branch protection flips to require the gate job, remove `ci.yml` (see 7.5) and delete this TODO line.

Adopt individually; update sections 7.3 / 7.4 when shipped.

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
