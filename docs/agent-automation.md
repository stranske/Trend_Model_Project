# Agent Automation & Telemetry Overview

_Last updated: 2025-09-12_

This document summarizes the GitHub Actions automation that powers agent assignment, labeling, formatting autofix, Codex bootstrap PR creation, and watchdog telemetry in this repository.

## High-Level Flow

```
Issue Labeled agent:copilot  ──▶ assign-to-agent.yml (issues) ──▶ GraphQL suggestActors → assign copilot-swe-agent
Issue Labeled agent:codex    ──▶ assign-to-agent.yml (issues) ──▶ sets needs_codex_bootstrap=true
                                                       │
                                                       ▼
                                   codex_bootstrap job creates draft PR, assigns, labels
                                                       │
PR Opened/Updated               ──▶ label-agent-prs.yml (pull_request_target) → idempotent labeling
                                                       │
PR (any)                        ──▶ autofix.yml (pull_request) → composite autofix action
                                                       │
Failing CI / Docker / Tests     ──▶ autofix-on-failure.yml (workflow_run) → composite autofix action
                                                       │
Issue / PR lacks Codex PR       ──▶ agent-watchdog.yml (schedule or dispatch) → state telemetry
```

## Key Workflows

### 1. `assign-to-agent.yml`
Triggers: `issues` (opened, labeled, reopened), `pull_request_target` (opened, labeled, reopened), manual `workflow_dispatch`.

Responsibilities:
- Resolve `agent:*` labels to registry entries (default + optional `.github/agents.json`).
- Assign Copilot via GraphQL (issues only; PR path posts breadcrumb reminder).
- Defer Codex bootstrap to secondary job via outputs (`needs_codex_bootstrap`).
- Assign generic agents via REST fallback.
- Write JSON artifact `agent_assignment.json` for telemetry / auditing.
- Bootstrap job (`codex_bootstrap`): creates/reuses branch `agents/codex-issue-<n>`, replicates issue title & body into PR, assigns actors, posts start command, drops marker file to ensure idempotency.

Outputs (first job):
- `needs_codex_bootstrap`: `true|false`
- `codex_issue`: issue number needing Codex PR
- `copilot_assigned`: whether Copilot was successfully assigned
- `generic_agents`: comma list of non-core agents assigned

Marker files:
- `agents/.codex-bootstrap-<issue>.json` prevents duplicate Codex PR creation.

### 2. `codex_bootstrap` (job inside `assign-to-agent.yml`)
Implemented via composite action: `.github/actions/codex-bootstrap`.

Responsibilities:
- Enforces PAT gating (exit 86) unless `CODEX_ALLOW_FALLBACK=true`.
- Optional network preflight with retries (see Variables below) to reduce transient API flakiness before performing branch/PR writes.
- Creates or reuses branch `agents/codex-issue-<n>` and marker file `agents/.codex-bootstrap-<n>.json`.
- Auto (default) mode: opens or reuses draft PR with replicated issue content.
- Manual mode: only branch + marker + instructions comment (user opens draft PR).
- Writes structured artifact `codex_bootstrap_summary.json` (always) for downstream inspection.
- Idempotent: re-run when marker exists emits summary without duplicating PR.

Composite inputs (selected):
| Input | Purpose | Default |
|-------|---------|---------|
| `issue` | Issue number to bootstrap | (required) |
| `service_bot_pat` | PAT for human-authored PR & comments | "" |
| `allow_fallback` | Permit fallback to `GITHUB_TOKEN` | false |
| `pr_mode` | `auto` or `manual` | auto |
| `codex_command` | Activation command (validated) | `codex: start` |
| `net_retry_attempts` | Preflight HTTP attempts | 1 |
| `net_retry_delay_s` | Delay between attempts | 2 |
| `fail_on_token_mismatch` | Fail if PAT present but author bot | unset |

### 3. `label-agent-prs.yml`
Trigger: `pull_request_target` (opened, synchronize, reopened).

Rationale for `pull_request_target`:
- Needs secret (PAT) for cross-fork labeling edge cases.
- No code checkout (reads event payload only) → mitigates secret exfil risk.
- Ensures labels applied before downstream workflows evaluate label predicates.

Idempotent: computes label delta and applies only missing labels.

### 4. `autofix.yml`
Trigger: standard `pull_request` events.

Logic:
- Skips drafts unless explicitly labeled `autofix`.
- Uses composite action `.github/actions/autofix` to install pinned versions and run ruff/black/isort/docformatter.
- Pushes changes only if formatter produced a diff (guard via `changed` output).

### 5. `autofix-on-failure.yml`
Trigger: `workflow_run` (CI, Docker, Lint, Tests) when conclusion != success.

Steps:
- Finds related open PR by head ref.
- Applies same composite autofix action (idempotent).
- Commits with canonical message `ci: autofix after failed checks` (loop guard in main autofix to avoid recursion).

### 6. `agent-watchdog.yml`
Purpose: Detect issues labeled for Codex where bootstrap PR not yet created OR gather fast telemetry.

Enhancements:
- Fast-mode: detects marker file presence to short-circuit deeper polling.
- Provides structured outputs (`state`) such as `FOUND` or `TIMEOUT`.
- Step summary enumerates any pending items.

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
| `agent_assignment.json` | `assign-to-agent.yml` | Auditable record of assignment decisions (inputs & outputs). |
| Step Summary tables | All major workflows | Human-readable status in Actions UI. |
| `state` output | `agent-watchdog.yml` | Programmatic detector for missing Codex bootstrap. |
| Marker files | Codex bootstrap job | Idempotency & external observable state via repo tree. |
| JSON summary comment | `assign-to-agent.yml` (new) | Machine-readable evergreen comment (issue/PR) with assignment + (later) bootstrap snapshot. |

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
| Autofix loop risk | Guard: skip when PR title starts with `ci: autofix`. |
| Formatting version drift | Shared `autofix-versions.env` ensures uniform versions. |

## Operational Playbook
1. Label issue with `agent:copilot` or `agent:codex`.
2. For Codex: expect draft PR within seconds; PR body mirrors issue content.
3. Review `agent_assignment.json` artifact if automation outcome unclear.
4. On failure to bootstrap: check issue comments for diagnostic message.
5. Use `workflow_dispatch` on `assign-to-agent.yml` for historical backfill.
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
