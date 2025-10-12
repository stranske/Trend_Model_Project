<!-- bootstrap for codex on issue #2462 -->

## Scope / Key Constraints
- Touch only the orchestration workflows involved in delegating to `reusable-70-agents.yml` (`.github/workflows/reusable-71-agents-dispatch.yml` and the reusable workflow itself) plus related documentation (`docs/ci_reuse.md`).
- Remove the disallowed `timeout-minutes` attribute from the caller job; GitHub Actions forbids timeouts on jobs that use another workflow.
- Reintroduce equivalent or tighter per-job timeouts inside `reusable-70-agents.yml`, keeping each long-running job within a 15–30 minute ceiling and preserving existing behaviour.
- Preserve workflow inputs, secrets pass-through, permissions, and job names so downstream automation continues to function.
- Updates must maintain YAML validity (passes `actionlint`/workflow schema) and align with existing repository conventions for documentation tone and formatting.

## Acceptance Criteria / Definition of Done
- `.github/workflows/reusable-71-agents-dispatch.yml` validates without “Unexpected value 'timeout-minutes'” or other structural errors when linted or parsed by GitHub Actions.
- `.github/workflows/reusable-70-agents.yml` defines explicit `timeout-minutes` values for every long-running job (readiness probe, preflight, diagnostic/bootstrap, keepalive variants, watchdog) within the agreed 15–30 minute window.
- Running the **Agents 70 Orchestrator** workflow via `workflow_dispatch` completes end-to-end, invoking the reusable workflow without YAML validation errors.
- Documentation in `docs/ci_reuse.md` explains where timeouts live (within the reusable workflow) and how to adjust them going forward.

## Initial Task Checklist
- [x] Inspect the existing `reusable-71-agents-dispatch.yml` job definition and remove the top-level `timeout-minutes` property from the `uses` job.
- [x] Audit `reusable-70-agents.yml` to confirm each long-running job has an explicit timeout; add or adjust values to stay within 15–30 minutes as appropriate.
- [x] Update `docs/ci_reuse.md` to describe the timeout placement and modification process.
- [x] Trigger or document the plan for a `workflow_dispatch` run of Agents 70 Orchestrator to verify the YAML loads successfully (include evidence or follow-up instructions). Documented manual dispatch steps in `docs/ci_reuse.md`; execution requires GitHub Actions UI access.
