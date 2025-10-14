# Issue #2560 — Orchestrator Workflow Consolidation Plan

## Scope & Key Constraints
- Limit changes to the GitHub Actions orchestrator workflows, focusing on `.github/workflows/agents-70-orchestrator.yml` and the reusable verification workflow it calls.
- Replace the sprawling `workflow_dispatch` input set with a single JSON payload (`params_json`) plus any essential toggles while preserving scheduled behavior.
- Keep downstream jobs consuming parameters exclusively through the resolve step outputs; avoid reading `github.event.inputs` elsewhere.
- Maintain compatibility with existing orchestrator defaults, ensuring runs without explicit `params_json` still succeed using sensible fallbacks.
- Follow GitHub Actions limitations (10 inputs per dispatch, 64KB JSON size) and avoid embedding secrets or repository-specific identifiers in defaults.
- Preserve manual run ergonomics by surfacing a concise parameters summary in the run output.

## Acceptance Criteria / Definition of Done
1. `.github/workflows/agents-70-orchestrator.yml` validates without "Invalid workflow file" errors and exposes only the `params_json` dispatch input (with optional top-level toggles if justified).
2. The “Resolve Parameters” step parses `params_json`, applies structured defaults, and publishes outputs consumed by all downstream jobs; no job reads `inputs.<legacy>` directly.
3. The verification leg invokes `agents-64-verify-agent-assignment.yml` via `workflow_call`, and the reusable workflow retains its own per-job timeouts.
4. A manual dispatch using `{ "enable_verify_issue":"true", "verify_issue_number":"1234" }` completes successfully and triggers the verification workflow at least once.
5. The orchestrator run summary renders a compact table (or equivalent) listing the effective parameters for auditing.
6. Evidence of the green manual run (URL + parameter summary) is attached to the source issue.

## Initial Task Checklist
- [ ] Audit `.github/workflows/agents-70-orchestrator.yml` to catalogue all current inputs, defaults, and consumption points.
- [ ] Draft the new `params_json` schema, including default constants for readiness agents, verify issue assignees, and optional feature toggles.
- [ ] Refactor the resolve step to parse `params_json`, merge defaults, and expose outputs for every downstream consumer.
- [ ] Update downstream jobs and reusable workflow invocations to consume the new outputs exclusively.
- [ ] Verify `.github/workflows/agents-64-verify-agent-assignment.yml` declares `on: workflow_call` and adjust invocation parameters/timeouts as needed.
- [ ] Exercise the workflow via `act` or dry-run checks to confirm syntax validity before manual dispatch.
- [ ] Perform the manual `workflow_dispatch` run with the provided payload, confirm success of all legs, and capture the run URL.
- [ ] Document the manual run evidence in the source issue and update repository artifacts if required (e.g., run summary screenshots or logs).
