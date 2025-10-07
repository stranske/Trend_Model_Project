# Agents Workflow Bootstrap Plan

## Scope & Key Constraints
- Reduce `workflow_dispatch` inputs in `.github/workflows/agents-consumer.yml` to stay within GitHub's 10 input limit by consolidating toggles into a single JSON payload (`params_json`).
- Preserve existing scheduling, concurrency, and downstream job wiring so that the reusable workflow `reuse-agents.yml` continues to receive all required flags.
- Ensure the consumer workflow parses the JSON payload safely (handle defaults for optional flags) without introducing dependencies on repository secrets for no-op or dry-run scenarios.
- Emit updated outputs (`issue_numbers_json`, `first_issue`) from the readiness step without breaking the Markdown summary table that downstream visibility relies on.
- Repair JSON handling in `.github/workflows/reuse-agents.yml` so expressions avoid unsupported concatenation operators and remain compatible with `fromJSON` usage.

## Acceptance Criteria / Definition of Done
1. `agents-consumer.yml` validates on GitHub (no workflow schema errors) and is manually dispatchable with ≤ 10 inputs.
2. Dispatch UI exposes a single `params_json` field, with documentation for a canonical payload example checked into the repository.
3. Consumer workflow parses `params_json` and passes all expected values to `reuse-agents.yml`, preserving default behaviours when keys are omitted.
4. Readiness step publishes both `issue_numbers_json` (JSON array string) and `first_issue` (stringified issue number) outputs while keeping the Markdown readiness report intact.
5. Bootstrap job pulls the first ready issue using either `first_issue` or `fromJSON(issue_numbers_json)[0]` without expression syntax errors.
6. `reuse-agents.yml` loads successfully, avoiding the previous `Unexpected symbol '+'` failure, and keeps watchdog/verification jobs callable when toggled on.
7. Optional: Workflow concurrency remains configured to prevent overlapping consumer runs.

## Initial Task Checklist
- [x] Audit `.github/workflows/agents-consumer.yml` inputs and map each existing flag into the planned `params_json` structure with default values.
- [x] Update the workflow to accept a single `params_json` input, parse it (e.g., via `fromJSON`) within the consumer job, and expose individual values for the reusable workflow call.
- [x] Adjust the readiness step to emit both JSON array and first-issue outputs while retaining the Markdown summary output.
- [x] Modify the bootstrap step to consume the new outputs when selecting the issue to pass into `reuse-agents.yml`.
- [x] Patch `.github/workflows/reuse-agents.yml` to remove invalid string concatenation in expressions, ensuring JSON parsing works with the new outputs.
- [x] Document an example `params_json` payload (see `.github/workflows/README.md`) so operators can easily trigger the workflow.
- [x] Smoke-test workflow syntax locally (pytest guard suite) and verify no secrets are required for dry-run paths.

## Verification Log

- 2025-03-04 – `pytest tests/test_workflow_agents_consolidation.py tests/test_workflow_naming.py`
  confirms the consumer/orchestrator/bridge workflows parse and document the
  consolidated `params_json` contract without re-introducing legacy dispatch
  inputs.
- README references for the JSON payload and operators' checklist are current
  as of this verification run.
