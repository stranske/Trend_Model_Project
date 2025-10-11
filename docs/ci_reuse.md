# Reusable CI & Automation Workflows

Issue #2190 leaves five reusable GitHub Actions workflows in this repository. They provide CI, autofix, and agent automation
building blocks that thin wrappers (or downstream repositories) can consume.

| Reusable Workflow | File | Purpose |
| ------------------ | ---- | ------- |
| Reusable CI | `.github/workflows/reusable-ci.yml` | Primary Python quality gate (lint, types, pytest, coverage). Used by Gate for Python 3.11/3.12.
| Python CI (legacy matrix) | `.github/workflows/reusable-90-ci-python.yml` | Matrix executor with optional coverage/metrics toggles retained for downstream callers.
| Legacy Python CI | `.github/workflows/reusable-94-legacy-ci-python.yml` | Compatibility contract for consumers still on the pre-WFv1 interface.
| Autofix | `.github/workflows/reusable-92-autofix.yml` | Formatting / lint autofix harness used by `maint-32-autofix.yml`.
| Agents Toolkit | `.github/workflows/reusable-70-agents.yml` | Readiness, Codex bootstrap, verification, and watchdog routines.
| Self-Test Matrix | `.github/workflows/reusable-99-selftest.yml` | Exercises the reusable CI executor across feature combinations.

## 1. Reusable CI (`reusable-ci.yml`)
Consumer example (excerpt from `pr-gate.yml`):

```yaml
jobs:
  core-tests-311:
    name: core tests (3.11)
    uses: ./.github/workflows/reusable-ci.yml
    with:
      python-version: '3.11'
      marker: "not quarantine and not slow"
```

Key inputs include the Python version and optional pytest marker expression. The reusable job installs dependencies, runs Ruff,
Mypy, and pytest with coverage, then uploads artifacts under the `coverage-<python>` naming convention.

## 2. Python CI (Legacy Matrix) (`reusable-90-ci-python.yml`)
Retained for downstream repositories that still depend on the Issue #2190 interface. Inputs expose matrix execution,
coverage/metrics toggles, and optional history tracking. Gate no longer calls this workflow directly but downstream consumers may
continue to do so until they migrate to `reusable-ci.yml`.

## 3. Autofix (`reusable-92-autofix.yml`)
Used by `maint-32-autofix.yml` to apply hygiene fixes once CI succeeds. Inputs gate behaviour behind opt-in labels and allow
custom commit prefixes. The composite enforces size/path heuristics before pushing changes with `SERVICE_BOT_PAT`.

## 4. Agents Toolkit (`reusable-70-agents.yml`)
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

## 5. Self-Test Matrix (`reusable-99-selftest.yml`)
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
