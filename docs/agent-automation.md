# Agent Automation & Telemetry Overview

_Last updated: 2026-10-12_

This document captures the trimmed agent automation surface that remains after Issue #2190. The GitHub Actions footprint now
consists of a single orchestrator workflow plus the reusable composite it consumes. Everything else that previously handled
label forwarding, watchdog wrappers, or Codex bootstrap fallbacks has been removed.

## High-Level Flow

```
Manual dispatch / hourly schedule ──▶ agents-70-orchestrator.yml
                                     │
                                     ├─ Readiness probes (GraphQL assignability)
                                     ├─ Optional Codex preflight diagnostics
                                     ├─ Optional issue verification (label + assignment parity)
                                     └─ Watchdog sweep for Codex bootstrap health
```

- No automatic label forwarding remains. Maintainers trigger the orchestrator directly from the Actions tab (manual
  `workflow_dispatch`) or allow the hourly schedule to run readiness + watchdog checks.
- Bootstrap PR creation, diagnostics, and stale issue escalation now live entirely inside `agents-70-orchestrator.yml` and the
  `reusable-70-agents.yml` composite it calls. Historical wrappers (`agents-41-assign*.yml`, `agents-42-watchdog.yml`, etc.) were
  deleted.

## Key Workflow

### `agents-70-orchestrator.yml`

- **Triggers:** `schedule` (top of every hour) and manual `workflow_dispatch` with curated inputs.
- **Inputs:** `enable_readiness`, `readiness_agents`, `require_all`, `enable_preflight`, `codex_user`,
  `enable_verify_issue`, `verify_issue_number`, `enable_watchdog`, `draft_pr`, plus an extensible `options_json` string for long
  tail toggles (currently `diagnostic_mode`, `readiness_custom_logins`, `codex_command_phrase`).
- **Behaviour:** delegates directly to `reusable-70-agents.yml`, which orchestrates readiness probes, Codex bootstrap, issue
  verification, and watchdog sweeps. The JSON options map is parsed via `fromJson()` so new flags can be layered without
  exploding the dispatch form beyond GitHub's 10-input limit.
- **Permissions:** retains `contents`, `pull-requests`, and `issues` write scopes to continue authoring Codex PRs or posting
  remediation comments.
- **Outputs:** inherits the reusable workflow's job summaries, watchdog tables, and readiness reports.

### Reusable Composite

`reusable-70-agents.yml` remains the single source of truth for agent automation logic:

- exposes a `workflow_call` interface so the orchestrator can exercise readiness, preflight, verification, and watchdog routines.
- keeps compatibility inputs such as `readiness_custom_logins`, `require_all`, `enable_preflight`, `enable_verify_issue`,
  `enable_watchdog`, and `draft_pr`.
- writes summarized Markdown + JSON artifacts for readiness probes and watchdog runs.

## Related Automation

While the agent wrappers were removed, maintenance automation still supports the broader workflow stack:

- `maint-32-autofix.yml` continues to follow the CI pipeline and apply low-risk fixes.
- `maint-30-post-ci-summary.yml` posts consolidated run summaries once `pr-10-ci-python.yml` and `pr-12-docker-smoke.yml` finish.
- `maint-33-check-failure-tracker.yml` opens or resolves CI failure issues based on those runs.

## Operational Playbook

1. Use the **Agents 70 Orchestrator** workflow to run readiness checks, Codex bootstrap diagnostics, or watchdog sweeps on demand.
2. Supply additional toggles via `options_json`, for example:
   ```json
   {
     "readiness_custom_logins": "my-bot,backup-bot",
     "diagnostic_mode": "full",
     "codex_command_phrase": "@codex start"
   }
   ```
3. Review the run summary for readiness tables, watchdog escalation indicators, and Codex bootstrap status.
4. Repeat manual dispatches as needed; scheduled runs provide hourly coverage for stale bootstrap detection.

## Security Considerations

- All sensitive operations continue to rely on `SERVICE_BOT_PAT` when available. The workflows gracefully fall back to
  `GITHUB_TOKEN` only when explicitly allowed by the repository variables.
- Inputs that toggle optional behaviour remain string-valued (`'true'` / `'false'`) to stay compatible with the reusable
  composite.

## Future Enhancements

- Extend `options_json` to cover any additional toggles without growing the dispatch form.
- Consider adding a lightweight CLI wrapper that posts curated `options_json` payloads for common scenarios.
- Monitor usage; if the hourly schedule proves redundant, convert it to manual-only to further reduce background noise.

For questions or updates, open an issue labeled `agent:codex` describing the desired change.
