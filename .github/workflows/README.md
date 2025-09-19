# Workflow & Agent Automation Quick Start (Issue #1204)

This guide enables a new maintainer to understand and operate the CI + agent automation stack in under 10 minutes.

---
## 1. Architecture Snapshot
Core layers:
- Reusable CI (`reuse-ci-python.yml`): tests, coverage, style.
- Reusable Autofix (`reuse-autofix.yml` + consumer): formatting & linting.
- Reusable Agents (`reuse-agents.yml` + consumer): modes for readiness, preflight, diagnostic, verify, watchdog, bootstrap.
- Governance & Health: `repo-health-self-check.yml`, labelers, dependency review, CodeQL.
- Path Labeling: `pr-path-labeler.yml` auto‑categorizes PRs.

Event Flow (conceptual):
1. PR opened → path labeler & agent labeler apply labels.
2. Labels + branch rules trigger CI / autofix / agent readiness.
3. Maintainer approves (CODEOWNERS) → low‑risk + `automerge` merges automatically.
4. Scheduled jobs (health, CodeQL) ensure hygiene.

---
## 2. Label Cheat Sheet
| Label | Purpose | Source |
|-------|---------|--------|
| `agent:codex` / `agent:copilot` | Classify automation origin PRs | Agent labeler heuristic |
| `from:codex` / `from:copilot` | Origin marker | Agent labeler |
| `risk:low` | Enables low‑friction auto‑merge path | Issue form defaults / agent labeler |
| `automerge` | Eligible for merge automation after checks | Issue form defaults / labeler |
| `codex-ready` | Signals agent bootstrap allowed | Issue templates |
| `type:bug` / `type:feature` | Issue taxonomy | Issue templates |
| Area labels (e.g. `backend`, `docs`, `tests`) | Scope classification | Path labeler |

---
## 3. Required Secrets & Variables
| Name | Type | Required | Purpose | Notes |
|------|------|----------|---------|-------|
| `SERVICE_BOT_PAT` | Secret | Recommended | Enables cross‑fork label/PR ops with stable identity | PAT with `repo` scope |
| `CODEX_ALLOW_FALLBACK` | Repo / Org Variable | Optional | Allow fallback to GITHUB_TOKEN when PAT absent | Set to `true` only temporarily |
| `AUTOMERGE_LABEL` | Variable | Optional | Override default `automerge` label name | Defaults inside workflows |
| `RISK_LABEL` | Variable | Optional | Override default `risk:low` label name | Keep consistent across docs |
| `AGENT_LABEL` / `AGENT_LABEL_ALT` | Variable | Optional | Customize agent classification labels | Paired with origin labels |
| `AUTOFIX_OPT_IN_LABEL` | Variable | Optional | Gate autofix in draft or certain repos | Falls back to `autofix` |

All other workflows operate with the default `GITHUB_TOKEN` under principle of least privilege.

---
## 4. Trigger Matrix
| Workflow | Trigger(s) | Key Inputs / Modes |
|----------|------------|--------------------|
| `reuse-ci-python.yml` | PR, push | Python matrix, coverage threshold |
| `reuse-autofix.yml` (consumer) | PR label / event | Formatting + lint patch apply |
| `reuse-agents.yml` (consumer) | `workflow_dispatch`, labels | readiness, preflight, diagnostic, verify_issue, watchdog, bootstrap |
| `repo-health-self-check.yml` | schedule, manual | Governance audit (labels, secrets, branch protection) |
| `pr-path-labeler.yml` | PR events | Path-based label sync |
| `label-agent-prs.yml` | PR (target) | Agent heuristics + baseline risk/automerge |
| `codeql.yml` | push, PR, schedule | Code scanning (Python) |
| `dependency-review.yml` | PR | Dependency diff & severity gate |

---
## 5. Adopting Reusable Workflows in Another Repository
Minimal example (in target repo):
```yaml
# .github/workflows/ci.yml
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
Autofix consumer:
```yaml
name: Autofix
on: [pull_request]
jobs:
  call:
    uses: stranske/Trend_Model_Project/.github/workflows/reuse-autofix.yml@phase-2-dev
```
Agents consumer (selected modes):
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
> Replace `phase-2-dev` with a tagged release ref when versioned.

---
## 6. Standard Onboarding (≈7 Minutes)
1. Create required labels: `automerge`, `risk:low`, `agent:codex`, `agent:copilot`, `codex-ready`.
2. Add optional area labels (backend, docs, tests, etc.).
3. Add secret `SERVICE_BOT_PAT` (or set `CODEX_ALLOW_FALLBACK=true` temporarily).
4. Configure Actions permissions: allow read+write for workflows.
5. Copy reusable workflow consumers (CI, autofix, agents) into repository.
6. Open a dummy PR → confirm path + agent labels applied.
7. Trigger agent readiness (if applicable) via `workflow_dispatch` selecting readiness/preflight.

---
## 7. Troubleshooting Pointers
| Symptom | Likely Cause | Reference |
|---------|--------------|-----------|
| No labels applied | Missing labeler workflow or insufficient permissions | `label-agent-prs.yml` |
| Agent bootstrap blocked (exit 86) | PAT missing & fallback disabled | `agent_codex_troubleshooting.md` |
| Autofix skipped | PR title matches autofix commit or opt-in label absent | Autofix consumer notes |
| Dependency review missing | Fork PR without diff or Action disabled | `dependency-review.yml` |
| CodeQL no alerts | First run pending or analysis still indexing | `codeql.yml` |

---
## 8. Extending
- Add quarantine job (planned) by extending `reuse-ci-python.yml` inputs.
- Introduce severity gating changes for dependency review (`fail-on-severity` tuning).
- Version reusables via annotated tags for external stability.

---
## 9. Related Deep-Dive Docs
| Topic | Doc |
|-------|-----|
| Reusable workflow design | `docs/ci_reuse.md` |
| Consolidation history | `docs/ci_reuse_consolidation_plan.md` |
| Agent automation modes | `docs/agent-automation.md` |
| Codex troubleshooting | `docs/agent_codex_troubleshooting.md` |
| Bootstrap verification scenarios | `docs/codex_bootstrap_verification.md` |
| Service bot facts | `docs/ops/codex-bootstrap-facts.md` |

---
## 10. Change Process
Submit a PR updating this README plus any workflow changes. Major workflow semantics should be noted in a short “Design Note” block inside the workflow file and linked here if impactful.

_Last updated: 2025-09-19 (implements Issue #1204)_
