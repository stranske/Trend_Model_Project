# Repo Ops Facts — Codex Bootstrap

_Last updated: 2026-10-12_

Issue #2190 collapsed the Codex automation surface to a single orchestrator. Issue #2650 reaffirmed that stance by removing the
last consumer shim and codifying the issue-bridge → orchestrator hand-off. This page captures the authoritative facts that
remain true after the cleanup.

## Branches and Events
- Default branch: `phase-2-dev`.
- Manual dispatches of **Agents 70 Orchestrator** run from the current default branch.
- The orchestrator continues to create a fresh branch for every bootstrap run; no existing branches are re-used.

## Trigger Labels
- Primary issue label: `agent:codex` (case-sensitive). Use it to document that an issue expects Codex handling even though the
  workflow is manually dispatched.
- Aliases such as `agents:codex` are no longer acted upon automatically but can remain for historical context.
- Preferred entry point: open the [Agent task issue template](https://github.com/stranske/Trend_Model_Project/issues/new?template=agent_task.yml)
  so the required `agents` and `agent:codex` labels are applied up front.
- The template change from Issue #2650 guarantees both labels are present, ensuring the Codex issue bridge triggers and then
  defers work to `agents-70-orchestrator.yml`.

## PR Hygiene
- Codex PRs remain non-draft by default.
- `@codex start` is posted on the PR body during bootstrap.
- Automation assigns `chatgpt-codex-connector`; additional assignees may be configured via repository variables.

## Tokens & Secrets
- Token precedence for authoring PRs: `OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`.
- Workflows avoid token switching inside step-level `if` statements; decisions are made once and forwarded via environment
  variables to the reusable composite.

## Active Workflows & Actions
- **Codex issue bridge:** `.github/workflows/agents-63-codex-issue-bridge.yml`
  - Reacts to `agent:codex` (and `agents:codex`) labels plus manual dispatch.
  - Creates or reuses Codex bootstrap branches/PRs and posts copyable issue snippets + `@codex start` instructions.
- **Orchestrator:** `.github/workflows/agents-70-orchestrator.yml`
  - 20-minute cron plus manual dispatch.
  - Inputs: readiness toggles, Codex preflight, watchdog controls, issue verification, `options_json` for extended flags.
  - Calls `.github/workflows/reusable-16-agents.yml` for the actual implementation.
- **Legacy consumer wrappers:** Former manual shims (`agents-61-consumer-compat.yml`, `agents-62-consumer.yml`) have been retired now that all automation flows through the orchestrator.
- **Reusable composite:** `.github/workflows/reusable-16-agents.yml`
  - Provides readiness probes, Codex bootstrap, verification, and watchdog jobs.
  - Exposes Markdown + JSON summaries for downstream tooling.
- **Composite action:** `.github/actions/codex-bootstrap-lite/action.yml`
  - Handles PAT selection, branch creation (`agents/codex-issue-<num>-<runid>`), marker files, PR authoring, and notification
    comments.

Legacy wrappers (`agents-41-assign*.yml`, `agents-42-watchdog.yml`, `agents-44-copilot-readiness.yml`, etc.) remain deleted as
part of Issue #2190; the dedicated Codex issue bridge was reinstated to restore the label-driven workflow.

## Quick Index

| Concern | File |
|---------|------|
| Agent orchestrator | [`agents-70-orchestrator.yml`](../../.github/workflows/agents-70-orchestrator.yml) |
| Reusable agent stack | [`reusable-16-agents.yml`](../../.github/workflows/reusable-16-agents.yml) |
| Codex bootstrap composite action | [`.github/actions/codex-bootstrap-lite`](../../.github/actions/codex-bootstrap-lite/action.yml) |
| Gate workflow | [`pr-00-gate.yml`](../../.github/workflows/pr-00-gate.yml) |
| Autofix follower | [`maint-46-post-ci.yml`](../../.github/workflows/maint-46-post-ci.yml) |
| Failure tracker | [`maint-46-post-ci.yml`](../../.github/workflows/maint-46-post-ci.yml) |

## Operational Notes
- Run the orchestrator manually to re-bootstrap an issue, perform readiness checks, or trigger watchdog sweeps.
- Use `options_json` to pass extended flags: diagnostic mode (`off`, `dry-run`, `full`), additional readiness logins, a custom
  Codex command phrase, or Codex bootstrap toggles (set `enable_bootstrap: true`
  and optionally `bootstrap_issues_label`).
- The orchestrator still honours `SERVICE_BOT_PAT` when creating PRs; provide the secret to avoid `github-actions[bot]` authorship.
- A local dry-run helper (`python tools/simulate_codex_bootstrap.py …`) reproduces the `Find Ready Issues` outputs to confirm the
  JSON payload parsed by `fromJson(...)` remains well-formed when labels resolve to multiple issues.

## Failure Modes
| Failure | Mitigation |
|---------|------------|
| PAT missing | Orchestrator fails fast unless fallback is explicitly allowed via repository variables. |
| Requested agent unavailable | Readiness probe flags missing accounts; set `require_all: 'true'` to make the run fail. |
| Watchdog timeout | Investigate the summary table emitted by the run and manually follow up on stale issues. |

Retain this document as the single source of truth for Codex bootstrap behaviour. Update it whenever inputs, schedules, or
security posture change.
