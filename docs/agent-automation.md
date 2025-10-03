# Agent Automation & Telemetry Overview

_Last updated: 2026-02-07_

This document summarizes the GitHub Actions automation that powers agent assignment, labeling, formatting autofix, Codex bootstrap PR creation, and watchdog telemetry in this repository.

## High-Level Flow

```
Issue labeled agent:copilot  ──▶ agents-41-assign.yml (wrapper)
                              │
Issue labeled agent:codex    ──▶ agents-41-assign.yml (wrapper)
                              │
                              ▼
                agents-41-assign-and-watch.yml orchestrates assignment + bootstrap + watchdog
                              │
                              ▼
                agents-42-watchdog.yml (wrapper) ──▶ delegates manual watchdog runs when needed
                              │
PR opened/updated            ──▶ label-agent-prs.yml (pull_request_target) → idempotent labeling
                              │
PR (any)                     ──▶ autofix-consumer.yml (pull_request) → composite autofix action
                              │
Failing CI / Docker / Tests  ──▶ autofix-on-failure.yml (workflow_run) → composite autofix action
```

## Key Workflows

### 1. `agents-41-assign-and-watch.yml`
Triggers: `workflow_dispatch`, scheduled sweep (`*/30 * * * *`). Label events arrive via the `agents-41-assign.yml` wrapper.

Responsibilities:
- Resolve incoming events (manual overrides, forwarded label/unlabel payloads, scheduled sweeps) and determine the required operation.
- Call `reusable-90-agents.yml` for readiness probes so assignment and stale sweeps share a single availability check path.
- Assign Copilot / Codex automation accounts, post trigger commands, and run Codex bootstrap via `.github/actions/codex-bootstrap-lite` with fallback to `codex-issue-bridge.yml` when required.
- Monitor for cross-referenced PRs (when requested) and emit ✅/⚠️ watchdog comments.
- Clear assignments when agent labels are removed and sweep for stale issues; ping active agents or escalate when they are unavailable.

Key outputs:
- `agent`, `issue`, `operation`, and `watchdog` metadata recorded in job summaries for observability.
- Stale sweep summary table indicating which issues were pinged vs escalated.

Bootstrap behaviour:
- Branch naming: `agents/codex-issue-<num>-<run-id>`.
- Marker file: `agents/codex-<issue>.md` ensures consistent tree content.
- PR body mirrors issue content with source link and `@codex start` kickoff comment.
- Assigns `chatgpt-codex-connector` (and helper bot where available), applies `agent:codex` label.

### 2. `agents-41-assign.yml`
Triggers: `issues: [labeled, unlabeled]`, `pull_request_target: [labeled]`, manual `workflow_dispatch`.

Responsibilities:
- Preserve the historical trigger surface while delegating all logic to `agents-41-assign-and-watch.yml`.
- Forward raw event payloads so the unified workflow can reconstruct context without duplicating detection code.

### 3. `agents-42-watchdog.yml`
Purpose: Maintain the legacy manual watchdog entry point while delegating work to the unified orchestrator.

Activation: Manual `workflow_dispatch` (or automated calls from older tooling) — forwards context with `mode: watch` into `agents-41-assign-and-watch.yml`.

Highlights:
- Interface-compatible with the previous direct watchdog run (issue, agent, timeout, expected PR inputs).
- Keeps historical links and documentation stable while centralising logic in the unified workflow.

### 4. `label-agent-prs.yml`
Trigger: `pull_request_target` (opened, synchronize, reopened).

Rationale for `pull_request_target`:
- Needs secret (PAT) for cross-fork labeling edge cases.
- No code checkout (reads event payload only) → mitigates secret exfil risk.
- Ensures labels applied before downstream workflows evaluate label predicates.

Idempotent: computes label delta and applies only missing labels.

### 5. `autofix.yml`
Trigger: `workflow_run` for the `CI` workflow (types: `completed`).

Jobs:
- **`context`** – Resolves the target PR, applies loop guard (skip when actor is `github-actions` _and_ head commit starts with `chore(autofix):`), and inspects changed files for safe globs and size limits.
- **`small-fixes`** – Runs when CI succeeded and the diff stays within safe heuristics (≤ `AUTOFIX_MAX_FILES`, ≤ `AUTOFIX_MAX_CHANGES`, all paths match curated globs). Uses the composite autofix action, commits as `chore(autofix): apply small fixes`, pushes with `SERVICE_BOT_PAT`, and emits fork patches when push is impossible.
- **`fix-failing-checks`** – Runs when CI failed but every failing job name contains lint/format/type keywords. Applies the composite action, commits `chore(autofix): fix failing checks`, uploads patches for forks, and labels the PR `needs-autofix-review` if no change was produced.

Other behavior:
- Same PAT guard for pushes (no fallback to `GITHUB_TOKEN`).
- Restores/updates `autofix:clean` vs `autofix:debt` labels based on residual diagnostics.
- Uploads summary sections so maintainers can see eligibility decisions directly from the run.

### 6. `cleanup-codex-bootstrap.yml`
Scheduled cleanup of stale `agents/codex-issue-*` branches beyond age threshold.

## Composite Action: `.github/actions/autofix`
Encapsulates tool installation and formatting logic:
- Loads pinned versions from `autofix-versions.env`.
- Runs ruff (fix), black, isort, docformatter.
- Emits `changed=true|false` output and summary table.

## Telemetry & Artifacts
| Artifact / Output | Source | Purpose |
|-------------------|--------|---------|
| `agent_assignment.json` | `agents-41-assign-and-watch.yml` | Auditable record of assignment decisions (inputs & outputs). |
| Step Summary tables | All major workflows | Human-readable status in Actions UI. |
| Watchdog summary | `agents-41-assign-and-watch.yml` | Programmatic detector for missing Codex bootstrap (success/timeout). |
| Marker files | Codex bootstrap job | Idempotency & external observable state via repo tree. |
| JSON summary comment | `agents-41-assign-and-watch.yml` (future) | Machine-readable evergreen comment (issue/PR) with assignment + bootstrap snapshot. |

## Security Posture
- Principle of least privilege: `contents: write` only in `codex_bootstrap`; base assignment job uses `contents: read`.
- `pull_request_target` restricted to a single labeler workflow with no code checkout.
- Secrets (PAT) only consumed where necessary (Codex bootstrap & optional labeler fallback). Graceful degradation to `GITHUB_TOKEN` otherwise.
- Idempotent operations minimize repeated side-effect risk.

### PAT vs Fallback Policy (Codex Bootstrap)

| Scenario | SERVICE_BOT_PAT | `CODEX_ALLOW_FALLBACK` | Result |
|----------|-----------------|------------------------|--------|
| Recommended | present | (ignored) | Human-authored draft PR + comments (preferred) |
| PAT missing, fallback allowed | absent | true | Bootstrap proceeds with `GITHUB_TOKEN` (PR authored by github-actions[bot]) |
| PAT missing, fallback disallowed (default) | absent | false / unset | Bootstrap job fails fast (exit 86) with explicit remediation comment |

Set the repository (or org) variable `CODEX_ALLOW_FALLBACK=true` only if you accept bot-authored Codex PRs temporarily. Long term, configure a PAT with `repo` scope and store it as `SERVICE_BOT_PAT` secret.

### Additional Environment / Repository Variables

| Variable | Effect |
|----------|--------|
| `CODEX_NET_RETRY_ATTEMPTS` | Sets composite preflight `net_retry_attempts` (ensure integer ≥1). |
| `CODEX_NET_RETRY_DELAY_S` | Delay between preflight attempts in seconds. |
| `CODEX_FAIL_ON_TOKEN_MISMATCH` | If non-empty, workflow fails when PAT supplied but PR author shows as `github-actions[bot]`. |
| `CODEX_PR_MODE` | Global default for Codex PR mode (`auto` / `manual`). |
| `CODEX_SUPPRESS_ACTIVATE` | When set, suppresses initial activation comment. |


## Failure Modes & Handling
| Failure | Mitigation |
|---------|------------|
| PAT missing for Codex bootstrap | Early fail (exit 86) unless fallback explicitly allowed; issue comment provides remediation. |
| Copilot not enabled (no `copilot-swe-agent`) | GraphQL result triggers explicit failure + breadcrumb guidance. |
| Duplicate Codex label events | Marker file short-circuits re-bootstrap. |
| Autofix loop risk | Guard: workflow_run follower + skip when GitHub Actions pushes a head commit whose subject starts with `chore(autofix):`. |
| Formatting version drift | Shared `autofix-versions.env` ensures uniform versions. |

## Operational Playbook
1. Label issue with `agent:copilot` or `agent:codex`.
2. For Codex: expect draft PR within seconds; PR body mirrors issue content.
3. Review `agent_assignment.json` artifact if automation outcome unclear.
4. On failure to bootstrap: check issue comments for diagnostic message.
5. Use `workflow_dispatch` on `agents-41-assign-and-watch.yml` (or the `agents-41-assign.yml` wrapper) for historical backfill.
6. Watchdog run (scheduled) should report `FOUND` for newly created bootstrap markers; investigate `TIMEOUT` states or escalations surfaced by the stale sweep.

## Extensibility Hooks
- Add new agents via `.github/agents.json` with `assignee`, `mention`, `aliases`.
- Additional formatting tools: extend composite action; maintain pinned versions list.
- Telemetry expansion: append keys to `agent_assignment.json` (backwards-compatible additions are fine).

## Future Improvements (Backlog)
- Consolidated dashboard artifact combining assignment + watchdog states over time.
- Metrics export (JSON Lines) for external observability platform ingestion.
- Slack / Teams notification step for bootstrap failures.
- Automated stale branch closure heuristics tied to issue closure events.
- Enrich JSON summary with watchdog scan deltas & retry command surface.
- Optional backoff jitter for network preflight.
- Composite action versioning & changelog automation.

---
For questions or updates to this design, open an issue labeled `agent:codex` or `agent:copilot` and describe the desired change.
