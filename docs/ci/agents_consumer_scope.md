# Agents Consumer Workflow – Planning Notes

## Scope and Key Constraints
- Update the `agents-consumer.yml` workflow to support safe scheduled execution by preventing overlapping runs, enforcing job-level timeouts, and limiting default actions to essential probes.
- Changes must preserve compatibility with the existing `reuse-agents.yml` reusable workflow that the consumer invokes.
- Opt-in behaviours (preflight checks and bootstrap routines) must remain disabled unless explicitly requested via `params_json` payloads to avoid unexpected automation.
- Documentation updates should be limited to explaining the new opt-in behaviour in the README without altering unrelated sections of the project documentation.

## Acceptance Criteria / Definition of Done
- The `agents-consumer.yml` workflow declares `concurrency: { group: agents-consumer, cancel-in-progress: true }` at the workflow root so only a single run is active at a time.
- Every job defined within `agents-consumer.yml` sets a `timeout-minutes` value to bound runtime and terminate stalled tasks automatically.
- The default execution path only runs watchdog and readiness probes; preflight and bootstrap steps trigger solely when the `params_json` input explicitly enables them.
- `README.md` (or the relevant workflow documentation) states that bootstrap and preflight activities are opt-in via `params_json` so operators understand the new defaults.

## Initial Task Checklist
1. Review the current `agents-consumer.yml` and the referenced `reuse-agents.yml` workflow to catalogue existing jobs, inputs, and defaults.
2. Add workflow-level concurrency control and verify it does not conflict with downstream reusable workflow behaviour.
3. Audit each job in `agents-consumer.yml`, adding or updating `timeout-minutes` values to meet reliability targets (e.g., under one hour per job).
4. Confirm the workflow inputs keep watchdog and readiness enabled by default while requiring explicit `params_json` flags for preflight and bootstrap.
5. Update the README (or workflow documentation) to note the opt-in nature of bootstrap runs and provide guidance on configuring `params_json`.
6. Run linting or workflow validation tools (such as `act` or `yamllint`, if available) to ensure the modified YAML is syntactically correct.
7. Open a pull request summarizing the concurrency, timeout, and documentation updates once local verification passes.
