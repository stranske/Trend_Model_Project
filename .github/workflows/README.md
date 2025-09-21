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
Agents (subset):
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
