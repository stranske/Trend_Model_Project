# Self-Test Workflow Manualization Plan (Issue #2496)

## Scope and Key Constraints
- **Workflow coverage**: All `.github/workflows/selftest-*.yml` (and related maintenance self-test variants) must be inventoried so the manual-only policy applies consistently, including any archived or auxiliary copies referenced by the docs.
- **Trigger restriction**: Every self-test workflow may only declare a `workflow_dispatch:` trigger; scheduled (`schedule:`), push, pull-request, or other repo-event triggers must be removed to prevent automatic execution.
- **Redundancy cleanup**: Duplicate or superseded self-test workflows should be deleted rather than merely disabled, ensuring the remaining catalog reflects the canonical set.
- **Documentation alignment**: `docs/ci/WORKFLOWS.md` (and any other workflow guides) must explain that self-test pipelines are reference examples that run solely on demand.
- **Historical traceability**: File renames or deletions should use `git mv` and Git history-friendly methods so reviewers can trace previous automation runs if needed.

## Acceptance Criteria / Definition of Done
1. All self-test workflow files trigger exclusively via `workflow_dispatch` and contain no other event triggers or schedules.
2. Duplicate or obsolete self-test workflows are removed, with the authoritative set residing under the updated `selftest-8X-*` naming band described in Issue #2496.
3. Documentation describing CI workflows (at minimum `docs/ci/WORKFLOWS.md`) includes a note clarifying self-tests are manual examples and not part of automated CI.
4. Repository search confirms no lingering references to deleted self-test workflows or to automated triggers for self-tests.
5. Required checks (notably `pr-00-gate`) succeed after the workflow adjustments, demonstrating no unintended CI regressions.

## Initial Task Checklist
- [x] Inventory all self-test workflows (active and archived) and map them to the target `selftest-8X-*` naming scheme.
- [x] Remove non-manual triggers from each remaining self-test workflow, leaving only a `workflow_dispatch` block.
- [x] Delete duplicate or deprecated self-test workflows once replacements are confirmed.
- [x] Update documentation (`docs/ci/WORKFLOWS.md` and related guides) with the manual-only clarification for self-tests.
- [x] Run repository-wide searches (`rg "selftest"`) to verify no stale references to removed files or automated triggers remain.
- [x] Push the branch to exercise the Gate workflow and confirm the required checks pass.
- [x] Perform a final review ensuring the workflow catalog and docs reflect the cleaned-up, manual-only self-test policy.
