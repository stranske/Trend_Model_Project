# Agent Automation & Telemetry Overview

_Last updated: 2026-02-07_

This document summarizes the GitHub Actions automation that powers agent assignment, labeling, formatting autofix, Codex bootstrap PR creation, and watchdog telemetry in this repository.

## High-Level Flow

```
Issue labeled agent:copilot  ──▶ assign-to-agents.yml ──▶ assigns Copilot + posts trigger (if PR)
Issue labeled agent:codex    ──▶ assign-to-agents.yml ──▶ bootstrap branch/PR via codex-bootstrap-lite
                                                        │
                                                        ▼
                                agent-watchdog.yml monitors for cross-referenced PR (7 min timeout)
                                                        │
PR opened/updated              ──▶ label-agent-prs.yml (pull_request_target) → idempotent labeling
                                                        │
PR (any)                       ──▶ autofix-consumer.yml (pull_request) → composite autofix action
                                                        │
Failing CI / Docker / Tests    ──▶ autofix-on-failure.yml (workflow_run) → composite autofix action
```

## Key Workflows

### 1. `assign-to-agents.yml`
Triggers: `issues: [labeled]`, `pull_request_target: [labeled]`, manual `workflow_dispatch`.

Responsibilities:
- Resolve `agent:*` labels to registry entries (default + optional `.github/agents.json`).
- Assign Copilot and other automation accounts via REST; post start command on PRs when missing.
- For Codex issues, call the composite `.github/actions/codex-bootstrap-lite` to create a branch/PR and post `@codex start`.
- Dispatch `agent-watchdog.yml` with context (issue number, expected PR, timeout) to confirm parity.
- Emit JSON assignment summary (future telemetry hook).

Key outputs:
- `agent` (resolved agent key) and `number` (issue/PR) via job summary.
- `watchdog_timeout` (minutes) + `started_at` timestamp used for watchdog dispatch.

Bootstrap behaviour:
- Branch naming: `agents/codex-issue-<num>-<run-id>`.
- Marker file: `agents/codex-<issue>.md` ensures consistent tree content.
- PR body mirrors issue content with source link and `@codex start` kickoff comment.
- Assigns `chatgpt-codex-connector` (and helper bot where available), applies `agent:codex` label.

### 3. `label-agent-prs.yml`
Trigger: `pull_request_target` (opened, synchronize, reopened).

Rationale for `pull_request_target`:
- Needs secret (PAT) for cross-fork labeling edge cases.
- No code checkout (reads event payload only) → mitigates secret exfil risk.
- Ensures labels applied before downstream workflows evaluate label predicates.

Idempotent: computes label delta and applies only missing labels.

### 4. `autofix.yml`
Trigger: `workflow_run` for the `CI` workflow (types: `completed`).

Jobs:
- **`context`** – Resolves the target PR, applies loop guard (skip when actor is `github-actions` _and_ head commit starts with `chore(autofix):`), and inspects changed files for safe globs and size limits.
- **`small-fixes`** – Runs when CI succeeded and the diff stays within safe heuristics (≤ `AUTOFIX_MAX_FILES`, ≤ `AUTOFIX_MAX_CHANGES`, all paths match curated globs). Uses the composite autofix action, commits as `chore(autofix): apply small fixes`, pushes with `SERVICE_BOT_PAT`, and emits fork patches when push is impossible.
- **`fix-failing-checks`** – Runs when CI failed but every failing job name contains lint/format/type keywords. Applies the composite action, commits `chore(autofix): fix failing checks`, uploads patches for forks, and labels the PR `needs-autofix-review` if no change was produced.

Other behavior:
- Same PAT guard for pushes (no fallback to `GITHUB_TOKEN`).
- Restores/updates `autofix:clean` vs `autofix:debt` labels based on residual diagnostics.
- Uploads summary sections so maintainers can see eligibility decisions directly from the run.

### 6. `agent-watchdog.yml`
Purpose: Verify that Codex issues produce a linked PR within the expected timeframe and surface actionable diagnostics when they do not.

Activation: Automatically dispatched by `assign-to-agents.yml`; can also be triggered manually via `workflow_dispatch` with custom timeout/PR inputs.

Highlights:
- Polls the issue timeline for cross-referenced PR events using the GitHub API (mockingbird preview).
- Posts ✅ success comment with PR link when found, or ⚠️ timeout comment if no PR appears before the deadline.
- Records elapsed time in the job summary and exposes outputs (`found`, `pr`, `elapsed_minutes`).

### 7. `cleanup-codex-bootstrap.yml`
Scheduled cleanup of stale `agents/codex-issue-*` branches beyond age threshold.

## Composite Action: `.github/actions/autofix`
Encapsulates tool installation and formatting logic:
- Loads pinned versions from `autofix-versions.env`.
- Runs ruff (fix), black, isort, docformatter.
- Emits `changed=true|false` output and summary table.

## Telemetry & Artifacts
| Artifact / Output | Source | Purpose |
|-------------------|--------|---------|
| `agent_assignment.json` | `assign-to-agents.yml` | Auditable record of assignment decisions (inputs & outputs). |
| Step Summary tables | All major workflows | Human-readable status in Actions UI. |
| Watchdog summary | `agent-watchdog.yml` | Programmatic detector for missing Codex bootstrap (success/timeout). |
| Marker files | Codex bootstrap job | Idempotency & external observable state via repo tree. |
| JSON summary comment | `assign-to-agents.yml` (future) | Machine-readable evergreen comment (issue/PR) with assignment + bootstrap snapshot. |

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
5. Use `workflow_dispatch` on `assign-to-agents.yml` for historical backfill.
6. Watchdog run (scheduled) should report `FOUND` for newly created bootstrap markers; investigate `TIMEOUT` states.

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
