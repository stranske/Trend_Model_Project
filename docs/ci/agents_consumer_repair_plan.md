# agents-61-consumer-compat.yml Repair / Decommission Plan

## Scope & Key Constraints
- **Workflow coverage** – Focus on `.github/workflows/agents-61-consumer-compat.yml` and the reusable it calls (`reusable-70-agents.yml`). Do not alter orchestrator (`agents-70-orchestrator.yml`) wiring unless a contract change is required by the repair.
- **Invocation model** – Consumer must either call the reusable correctly via `workflow_call` semantics or be downgraded to `workflow_dispatch`-only usage with other triggers removed.
- **Schema compliance** – Any reusable workflow referenced must expose a `workflow_call` block with required `inputs` / `secrets`; consumers must supply those via the `uses` job pattern (`jobs.<id>.uses` and `with:` / `secrets:` blocks).
- **Concurrency** – Introduce or preserve a `concurrency:` group to avoid multiple overlapping consumer runs while debugging.
- **Telemetry parity** – Preserve existing summary output / annotations produced by the reusable. If the consumer is disabled, document equivalent manual entry points.
- **Rollback safety** – Avoid breaking other workflows. Changes should be limited to the consumer and reusable definitions plus associated documentation.

## Acceptance Criteria / Definition of Done
1. **Successful dispatch** – Manual `workflow_dispatch` runs of `agents-61-consumer-compat.yml` complete without schema errors. If downgraded, only `workflow_dispatch` remains as a trigger.
2. **Reusable contract** – `.github/workflows/reusable-70-agents.yml` contains a valid `on: workflow_call` definition with named `inputs` (and defaults where appropriate). Inputs/secrets consumed inside the reusable match those forwarded by the consumer.
3. **Consumer wiring** – `agents-61-consumer-compat.yml` invokes the reusable using a `jobs.<name>.uses: ./.github/workflows/reusable-70-agents.yml` block, forwarding `with:` parameters and `secrets: inherit` (or explicit mapping) as needed.
4. **Concurrency guard** – A `concurrency:` block exists on the consumer workflow referencing a stable group name (e.g., `agents-61-consumer-compat`).
5. **Trigger hygiene** – No `schedule:` or push/pull triggers remain if the workflow is intentionally constrained to manual execution.
6. **Validation window** – After merge, the workflow list should remain free of "Invalid workflow file" errors for at least 48 hours (monitored via Actions UI).
7. **Documentation update** – Reference material (e.g., `docs/ci/WORKFLOWS.md` or dedicated agent docs) notes the repaired trigger pattern and manual run guidance.

## Initial Task Checklist
- [ ] Inspect current Actions run failure logs for `.github/workflows/agents-61-consumer-compat.yml` to capture exact schema complaints.
- [ ] Audit `.github/workflows/reusable-70-agents.yml` to confirm or add `workflow_call` definition and enumerate required inputs/secrets.
- [ ] Update the reusable to declare defaults / required flags matching the consumer's expectations.
- [ ] Rewrite `agents-61-consumer-compat.yml` jobs section to use the reusable via `uses` syntax, forwarding `with:` parameters and `secrets: inherit`.
- [ ] Add / verify `concurrency:` block on the consumer workflow.
- [ ] Review triggers; remove `schedule:` and other events if retiring automation to manual-only.
- [ ] Run `actionlint` or equivalent schema validation locally / via CI to catch structural errors.
- [ ] Update workflow documentation (e.g., `docs/ci/WORKFLOWS.md`) to reflect the repaired or restricted consumer behaviour.
- [ ] Prepare follow-up monitoring plan to confirm no "Invalid workflow" alerts after deployment.
