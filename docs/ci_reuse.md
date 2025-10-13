# Reusable CI & Automation Workflows

Issues #2190 and #2466 consolidated the GitHub Actions surface into four
reusable composites plus a manual self-test. These building blocks underpin the
Gate workflow, maintenance jobs, and Codex automation. Treat the orchestrator as
the single entry point for agents; the numbered and legacy consumers exist only
as deprecated compatibility shims for callers that still emit a `params_json`
blob.

| Reusable Workflow | File | Purpose |
| ------------------ | ---- | ------- |
| Reusable CI | `.github/workflows/reusable-10-ci-python.yml` | Primary Python quality gate (lint, types, pytest, coverage). Used by Gate for Python 3.11/3.12. |
| Reusable Docker Smoke | `.github/workflows/reusable-12-ci-docker.yml` | Docker build + smoke test harness consumed by Gate and downstream callers. |
| Autofix | `.github/workflows/reusable-92-autofix.yml` | Formatting / lint autofix composite invoked by `autofix.yml` and `maint-30-post-ci.yml`. |
| Agents Toolkit | `.github/workflows/reusable-70-agents.yml` | Readiness, Codex bootstrap, diagnostics, verification, keepalive, and watchdog routines dispatched exclusively through the orchestrator. |
| Selftest 81 Reusable CI | `.github/workflows/selftest-81-reusable-ci.yml` | Manual matrix that exercises the reusable CI executor across documented feature toggles. |

## 1. Reusable CI (`reusable-10-ci-python.yml`)
Consumer example (excerpt from `pr-00-gate.yml`):

```yaml
jobs:
  core-tests-311:
    name: core tests (3.11)
    uses: ./.github/workflows/reusable-10-ci-python.yml
    with:
      python-version: '3.11'
      marker: "not quarantine and not slow"
```

Key inputs include the Python version and optional pytest marker expression. The
reusable job installs dependencies, runs Ruff, Mypy, and pytest with coverage,
then uploads artifacts under the `coverage-<python>` naming convention.

## 2. Reusable Docker Smoke (`reusable-12-ci-docker.yml`)
Gate calls this composite to build the Docker image and run the smoke-test
command. Downstream repositories can reuse it directly:

```yaml
jobs:
  docker-smoke:
    uses: stranske/Trend_Model_Project/.github/workflows/reusable-12-ci-docker.yml@phase-2-dev
```

No inputs are required; extend by forking the workflow and layering additional
steps if your project needs extra smoke assertions.

## 3. Autofix (`reusable-92-autofix.yml`)
Used by `autofix.yml` and `maint-30-post-ci.yml` to apply hygiene fixes once CI
succeeds. Inputs gate behaviour behind opt-in labels and allow custom commit
prefixes. The composite enforces size/path heuristics before pushing changes
with `SERVICE_BOT_PAT`.

## 4. Agents Toolkit (`reusable-70-agents.yml`)
Exposes the agent automation stack as a reusable component. All top-level
automation calls flow through **Agents 70 Orchestrator**, which normalises the
inputs and forwards them here. Dispatch the orchestrator either via the Actions
UI or by posting the `params_json` payload shown in
[docs/ci/WORKFLOWS.md](ci/WORKFLOWS.md#manual-orchestrator-dispatch)
(`gh workflow run agents-70-orchestrator.yml --raw-field params_json="$(cat orchestrator.json)"`
or the equivalent REST call using `curl -X POST ... '{"ref":"phase-2-dev","inputs":{"params_json":"..."}}'`).

Export `GITHUB_TOKEN` to a PAT or workflow token with `workflow` scope before dispatching via CLI or REST.

Example orchestrator snippet:

```yaml
jobs:
  orchestrate:
    uses: ./.github/workflows/reusable-70-agents.yml
    with:
      enable_readiness: ${{ inputs.enable_readiness || 'false' }}
      readiness_agents: ${{ inputs.readiness_agents || 'copilot,codex' }}
      enable_preflight: ${{ inputs.enable_preflight || 'false' }}
      enable_bootstrap: ${{ inputs.enable_bootstrap || 'false' }}
      bootstrap_issues_label: ${{ fromJson(inputs.options_json || '{}').bootstrap_issues_label || 'agent:codex' }}
      options_json: ${{ inputs.options_json || '{}' }}
```

Timeouts live inside the reusable workflow so the orchestrator avoids invalid
syntax. Each automation path has a bound sized to its typical runtime plus
headroom (readiness/preflight: 15 minutes, diagnostics: 20 minutes, bootstrap:
30 minutes, keepalive: 25 minutes).

## 5. Selftest 81 Reusable CI (`selftest-81-reusable-ci.yml`)
Runs the matrix that validates the reusable CI executor across feature
combinations (coverage delta, soft gate, metrics, history, classification). The
workflow triggers only on `workflow_dispatch`, keeping Actions history quiet
until a maintainer requests a run.

## Adoption Notes
1. Reference the files directly via `uses: stranske/Trend_Model_Project/.github/workflows/<file>@phase-2-dev` in external repos.
2. Pin versions or branch references explicitly; do not rely on floating defaults.
3. When adopting the agents toolkit, point automation at `agents-70-orchestrator.yml`. The numbered/legacy consumers exist only
   as deprecated shims for `params_json` payloads.

## Customisation Points
| Area | How to Extend | Notes |
| ---- | ------------- | ----- |
| Coverage reporting | Chain an additional job that depends on the reusable CI job to upload coverage artifacts. | Keep job IDs stable when referencing outputs. |
| Autofix heuristics | Update `autofix.yml` or `maint-30-post-ci.yml` to widen size limits or adjust glob filters. | Avoid editing the reusable composite unless behaviour must change globally. |
| Agents options | Provide extra keys inside `params_json` (and embed `options_json` when structured overrides are required) and update the reusable workflow to honour them. | Remember GitHub only supports 10 dispatch inputs; keep new flags in JSON. |

## Security & Permissions
- CI workflows default to `permissions: contents: read`; escalate only when artifacts require elevated scopes.
- Autofix pushes require `SERVICE_BOT_PAT`; keep fallback disabled unless intentionally allowing `github-actions[bot]` commits.
- Agents automation exercises repository write scopes and continues to fail fast if secrets are missing. The orchestrator honours
  PAT priority (`OWNER_PR_PAT` → `SERVICE_BOT_PAT` → `GITHUB_TOKEN`).

Keep this document aligned with the final workflow roster; update it whenever
inputs or defaults change.
