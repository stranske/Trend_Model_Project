# Repo Ops Facts — Codex Bootstrap

_Last updated: 2026-10-12_

Issue #2190 collapsed the Codex automation surface to a single orchestrator. This page captures the authoritative facts that
remain true after the cleanup.

## Branches and Events
- Default branch: `phase-2-dev`.
- Manual dispatches of **Agents 70 Orchestrator** run from the current default branch.
- The orchestrator continues to create a fresh branch for every bootstrap run; no existing branches are re-used.

## Trigger Labels
- Primary issue label: `agent:codex` (case-sensitive). Use it to document that an issue expects Codex handling even though the
  workflow is manually dispatched.
- Aliases such as `agents:codex` are no longer acted upon automatically but can remain for historical context.

## PR Hygiene
- Codex PRs remain non-draft by default.
- `@codex start` is posted on the PR body during bootstrap.
- Automation assigns `chatgpt-codex-connector`; additional assignees may be configured via repository variables.

## Tokens & Secrets
- Token precedence for authoring PRs: `OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`.
- Workflows avoid token switching inside step-level `if` statements; decisions are made once and forwarded via environment
  variables to the reusable composite.

## Active Workflows & Actions
- **Orchestrator:** `.github/workflows/agents-70-orchestrator.yml`
  - Hourly schedule + manual dispatch.
  - Inputs: readiness toggles, Codex preflight, watchdog controls, issue verification, `options_json` for extended flags.
  - Calls `.github/workflows/reusable-70-agents.yml` for the actual implementation.
- **Reusable composite:** `.github/workflows/reusable-70-agents.yml`
  - Provides readiness probes, Codex bootstrap, verification, and watchdog jobs.
  - Exposes Markdown + JSON summaries for downstream tooling.
- **Composite action:** `.github/actions/codex-bootstrap-lite/action.yml`
  - Handles PAT selection, branch creation (`agents/codex-issue-<num>-<runid>`), marker files, PR authoring, and notification
    comments.

All previous wrappers (`agents-41-assign*.yml`, `agents-42-watchdog.yml`, `agents-43-codex-issue-bridge.yml`, etc.) were deleted.

## Quick Index

| Concern | File |
|---------|------|
| Agent orchestrator | [`agents-70-orchestrator.yml`](../../.github/workflows/agents-70-orchestrator.yml) |
| Reusable agent stack | [`reusable-70-agents.yml`](../../.github/workflows/reusable-70-agents.yml) |
| Codex bootstrap composite action | [`.github/actions/codex-bootstrap-lite`](../../.github/actions/codex-bootstrap-lite/action.yml) |
| CI wrapper | [`pr-10-ci-python.yml`](../../.github/workflows/pr-10-ci-python.yml) |
| Docker smoke | [`pr-12-docker-smoke.yml`](../../.github/workflows/pr-12-docker-smoke.yml) |
| Autofix follower | [`maint-32-autofix.yml`](../../.github/workflows/maint-32-autofix.yml) |
| Failure tracker | [`maint-33-check-failure-tracker.yml`](../../.github/workflows/maint-33-check-failure-tracker.yml) |

## Operational Notes
- Run the orchestrator manually to re-bootstrap an issue, perform readiness checks, or trigger watchdog sweeps.
- Use `options_json` to pass extended flags: diagnostic mode (`off`, `dry-run`, `full`), additional readiness logins, or a custom
  Codex command phrase.
- The orchestrator still honours `SERVICE_BOT_PAT` when creating PRs; provide the secret to avoid `github-actions[bot]` authorship.

## Failure Modes
| Failure | Mitigation |
|---------|------------|
| PAT missing | Orchestrator fails fast unless fallback is explicitly allowed via repository variables. |
| Requested agent unavailable | Readiness probe flags missing accounts; set `require_all: 'true'` to make the run fail. |
| Watchdog timeout | Investigate the summary table emitted by the run and manually follow up on stale issues. |

Retain this document as the single source of truth for Codex bootstrap behaviour. Update it whenever inputs, schedules, or
security posture change.
