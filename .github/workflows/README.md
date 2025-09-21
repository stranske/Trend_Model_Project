# Workflow & Agent Automation Quick Start (Issue #1204)

This guide enables a new maintainer to operate the CI + agent automation stack in under 10 minutes.

---
## 1. Architecture Snapshot
Core layers:
- Reusable CI (`reuse-ci-python.yml`): tests, coverage, style.
- Reusable Autofix (`reuse-autofix.yml` + consumer): formatting & linting.
- Reusable Agents (`reuse-agents.yml` + consumer): readiness, preflight, diagnostic, verify, watchdog, bootstrap.
- Governance & Health: `repo-health-self-check.yml`, labelers, dependency review, CodeQL.
- Path Labeling: `pr-path-labeler.yml` auto-categorizes PRs.

### 1.1 Current CI Topology (Issue #1351)
The CI stack now runs in distinct lanes so each concern can evolve independently:

| Lane | Workflow(s) | Purpose | Required Status Today | Future Plan |
|------|-------------|---------|-----------------------|-------------|
| Core test/coverage | `reusable-ci-python.yml` (consumed by `ci.yml`) | Matrix tests, coverage report, style gates | Wrapper job "CI" (legacy) + gate job | Make gate job the only required check once stable |
| Gate aggregation | `reusable-ci-python.yml` job: `gate / all-required-green` | Ensures upstream jobs passed (single source of truth) | Secondary (not yet sole required) | Will replace wrapper after burn‑in |
| Coverage soft gate | `coverage_soft_gate` job (opt‑in) | Posts coverage & hotspots without failing builds | Disabled unless input `enable-soft-gate` true | Remains advisory; can be promoted later |
| Universal logs | `logs_summary` job | Adds per‑job log table to run summary | Not required | Always-on helper |
| Autofix lane | `reuse-autofix.yml` | Formatting, linting autofix patch | Not required | Remains optional |
| Codex bootstrap | `codex-issue-bridge.yml` (+ verify & preflight) | Converts issues into branches/PRs | Not required | Hardens with additional diagnostics |

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
| `reuse-autofix.yml` | PR events | Formatting patch |
| `reuse-agents.yml` | dispatch, labels | All agent modes |
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
Autofix:
```yaml
name: Autofix
on: [pull_request]
jobs:
  call:
    uses: stranske/Trend_Model_Project/.github/workflows/reuse-autofix.yml@phase-2-dev
```
Autofix commits always use the `chore(autofix):` prefix. When a run is triggered by `github-actions`, the reusable workflow
inspects the latest commit message and short-circuits if it already begins with that prefix. This guard stops autofix pushes
from triggering another autofix loop.

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
Autofix commits use the canonical prefix `ci: autofix` (e.g. `ci: autofix formatting/lint`).
Loop prevention is achieved via three layers:
1. Reusable Autofix job `if:` excludes automation actors (`github-actions`, `github-actions[bot]`).
2. Downstream autofix / failure handler workflows detect prior commits whose subject starts with `ci: autofix` and short‑circuit to avoid re‑trigger storms.
3. Commit message pattern is centralized through the `commit_prefix` input (default `ci: autofix`).

Result: Each human push gets at most one autofix patch sequence; autofix commits do not recursively spawn new autofix runs. Original issue suggested `chore(autofix):`; project standardized on `ci: autofix` for CI-related automation consistency.

---
## 7.2 Codex Kickoff Flow (Issue #1351)
End‑to‑end lifecycle for automation bootstrapped contributions:
1. Maintainer opens Issue with label `codex-ready` (and optional spec details).
2. `codex-issue-bridge.yml` triggers (label or manual dispatch) and resolves desired PR draft state via `codex_pr_draft` input (default: non‑draft).
3. Workflow creates a branch (naming convention: sanitized issue title / id) and an associated PR, posting a kickoff comment outlining next steps for the agent.
4. Subsequent agent workflows (`reuse-agents.yml` verify / diagnostic) run against that PR.
5. When automation pushes commits, path labelers & readiness jobs re-evaluate.
Troubleshooting: If branch/PR not created, verify the label `codex-ready`, permissions for `GITHUB_TOKEN` (write), and absence of conflicting existing branch name.

---
## 7.3 Coverage Soft Gate (Issue #1351)
Purpose: Provide early visibility of coverage / hotspot data without failing PRs.

Activation (consumer of `reusable-ci-python.yml`):
```yaml
with:
  enable-soft-gate: 'true'
```
Outputs:
- Run Summary section: "Coverage Soft Gate" with overall % (avg, worst) and hotspot list.
- Artifacts: `coverage.xml` (raw), any generated per-lane reports. (Trend artifact optional – see future enhancements.)
- Canonical Issue updates: If configured, the job appends run metrics to a designated coverage tracking Issue (see workflow inputs `coverage_issue_number`).

Behavior: Non‑blocking (always succeeds). If parsing errors occur, job emits a warning and skips posting instead of failing.
Hotspots: Derived by scanning per‑file coverage under threshold; sorted descending by uncovered lines.

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
| Coverage trend artifact (JSON) | Planned | Would store last N run stats for trend charting |
| Coverage trend history (NDJSON) | Implemented | `coverage-trend-history` artifact accumulates per-run records |
| Centralized autofix commit prefix constant | Planned | Reduce drift; single env var reused across workflows |
| Failing test count in logs summary | Planned | Surface # of failing test jobs inline |

TODO (wrapper removal): After branch protection flips to require the gate job, remove `ci.yml` (see 7.5) and delete this TODO line.

Adopt individually; update sections 7.3 / 7.4 when shipped.

---
_Addendum (Issue #1351): CI topology, kickoff flow, soft gate, logs summary, and migration plan documented. Wrapper removal pending future protection flip._

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
