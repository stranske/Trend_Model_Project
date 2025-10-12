# Selftest 81 Reusable CI Workflow Plan

## Scope and Key Constraints
- Replace step-level `uses` invocations in `.github/workflows/selftest-81-reusable-ci.yml` with job-level `jobs.<id>.uses` calls targeting `.github/workflows/reusable-10-ci-python.yml`.
- Preserve the workflow's opt-in nature (manual `workflow_dispatch` and/or label trigger) while keeping existing inputs or labels functional.
- Model the scenario execution as a matrix-driven reusable workflow job (`jobs.scenario`) that forwards scenario-specific inputs via `strategy.matrix.include` entries.
- Maintain or improve aggregation logic that collates scenario results, handles required artifacts, and exposes outputs to downstream jobs without breaking current consumers.
- Avoid introducing dependencies on unavailable GitHub-hosted runners or secrets; stay compatible with the repository's supported runner pool.
- Keep naming conventions, permissions, and environment protections aligned with repository workflow standards (see `docs/ci_reuse.md`).

## Acceptance Criteria / Definition of Done
1. `.github/workflows/selftest-81-reusable-ci.yml` dispatches reusable scenarios through a `jobs.scenario.uses` job referencing `reusable-10-ci-python.yml` with a clearly defined matrix of at least two scenarios.
2. Each matrix entry maps required inputs/needs (e.g., scenario identifier, environment, feature toggles) so the reusable workflow executes successfully for every scenario.
3. Aggregator job (`jobs.aggregate`) consumes the matrix job outputs via `needs`, produces a concise summary (artifacts/log output), and validates the presence of expected artifacts from each scenario run.
4. Workflow remains manually triggerable (e.g., `workflow_dispatch` or label-based trigger), and default branch protections are unchanged.
5. Documentation/comments within the workflow explain how to add/remove scenarios and how outputs are propagated to the aggregate job.
6. CI dry-run or linting (if available) confirms the workflow syntax is valid (e.g., `act` or `workflow-lint` if part of repo tooling).

## Initial Task Checklist
- [x] Inspect current `.github/workflows/selftest-81-reusable-ci.yml` to capture existing triggers, inputs, scenario definitions, and artifact handling.
- [x] Review `.github/workflows/reusable-10-ci-python.yml` to enumerate required inputs, outputs, and permissions for job-level reuse.
- [x] Draft the new `jobs.scenario` matrix structure, mapping per-scenario inputs to the reusable workflow.
- [x] Update the workflow file to replace step-level `uses` calls with `jobs.scenario.uses`, wiring matrix-provided values into the reusable workflow inputs.
- [x] Adjust aggregator job to consume `needs.scenario.outputs`, ensure artifact checks operate on matrix results, and refine summary messaging.
- [x] Add or update inline documentation within the workflow describing the matrix configuration and aggregation expectations.
- [x] Execute available workflow validation tooling (e.g., `workflow-lint`, `act` dry run) or perform manual syntax validation.
- [x] Capture follow-up notes or TODOs if additional automation or secrets configuration is required outside the workflow file.
