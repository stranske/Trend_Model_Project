# Reusable CI & Automation Workflows

Issue #2190 leaves five reusable GitHub Actions workflows in this repository. They provide CI, autofix, and agent automation
building blocks that thin wrappers (or downstream repositories) can consume.

| Reusable Workflow | File | Purpose |
| ------------------ | ---- | ------- |
| Python CI | `.github/workflows/reusable-90-ci-python.yml` | Tests + coverage gate + optional feature flags.
| Legacy Python CI | `.github/workflows/reusable-94-legacy-ci-python.yml` | Compatibility contract for consumers still on the pre-WFv1 interface.
| Autofix | `.github/workflows/reusable-92-autofix.yml` | Formatting / lint autofix harness used by `maint-32-autofix.yml`.
| Agents Toolkit | `.github/workflows/reusable-70-agents.yml` | Readiness, Codex bootstrap, verification, and watchdog routines.
| Self-Test Matrix | `.github/workflows/reusable-99-selftest.yml` | Exercises the reusable CI executor across feature combinations.

## 1. Python CI (`reusable-90-ci-python.yml`)
Consumer example:

```yaml
name: PR 10 CI Python
on:
  pull_request:
  push:
    branches: [ phase-2-dev ]

jobs:
  core:
    uses: ./.github/workflows/reusable-90-ci-python.yml
    with:
      python-versions: '["3.11"]'
      enable-metrics: 'true'
      enable-history: 'true'
```

Key inputs include optional coverage gates, history/metrics toggles, and the Python version matrix. The workflow emits gate,
coverage, and summary jobs that downstream consumers can depend upon.

## 2. Autofix (`reusable-92-autofix.yml`)
Used by `maint-32-autofix.yml` to apply hygiene fixes once CI succeeds. Inputs gate behaviour behind opt-in labels and allow
custom commit prefixes. The composite enforces size/path heuristics before pushing changes with `SERVICE_BOT_PAT`.

## 3. Agents Toolkit (`reusable-70-agents.yml`)
Exposes the agent automation stack as a reusable component. Inputs include readiness toggles, optional Codex preflight, issue
verification, watchdog control, and convenience flags such as `readiness_custom_logins`.

Example consumer snippet:

```yaml
jobs:
  orchestrate:
    uses: ./.github/workflows/reusable-70-agents.yml
    with:
      enable_readiness: 'true'
      readiness_agents: 'copilot,codex'
      require_all: 'true'
      enable_preflight: 'true'
      enable_watchdog: 'true'
      draft_pr: 'false'
```

The caller may also pass `options_json` (in the orchestration workflow) to layer additional toggles without exceeding GitHub's
input limit.

## 4. Self-Test Matrix (`reusable-99-selftest.yml`)
Exposes the matrix that validates the reusable CI executor across feature combinations (coverage delta, soft gate, metrics, history, classification). The workflow no longer declares its own cron or manual triggers; `maint-90-selftest.yml` is the thin caller for ad-hoc dispatches or weekly checks.

## Adoption Notes
1. Reference the files directly via `uses: stranske/Trend_Model_Project/.github/workflows/<file>@phase-2-dev` in external repos.
2. Pin versions or branch references explicitly; do not rely on floating defaults.
3. When adopting the agents toolkit, review the security posture—supply `SERVICE_BOT_PAT` and configure repository variables for
   fallback behaviour.

## Customisation Points
| Area | How to Extend | Notes |
| ---- | ------------- | ----- |
| Coverage reporting | Chain an additional job that depends on the reusable CI job to upload coverage artifacts. | Keep job IDs stable when referencing outputs. |
| Autofix heuristics | Update `maint-32-autofix.yml` to widen size limits or adjust glob filters. | Avoid editing the reusable composite unless behaviour must change globally. |
| Agents options | Provide extra keys inside `options_json` and update the reusable workflow to honour them. | Remember GitHub only supports 10 dispatch inputs; keep new flags in JSON. |

## Security & Permissions
- CI workflows default to `permissions: contents: read`; escalate only when artifacts require elevated scopes.
- Autofix pushes require `SERVICE_BOT_PAT`; keep fallback disabled unless intentionally allowing `github-actions[bot]` commits.
- Agents automation exercises repository write scopes and will continue to fail fast if secrets are missing.

Keep this document aligned with the final workflow roster; update it whenever inputs or defaults change.
