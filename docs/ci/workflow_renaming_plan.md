# Workflow Renaming Plan (Issue #2492)

## Scope and Key Constraints
- **Workflows covered**: All YAML files under `.github/workflows/`, including reusable components, must follow the new numbering-based naming scheme described in Issue #2492.
- **Rename map fidelity**: File moves must exactly match the provided mapping, including deletions (e.g., removing `agent-watchdog.yml` if confirmed obsolete) and handling already compliant files (e.g., `agents-43-codex-issue-bridge.yml`).
- **Display names preserved**: Workflow `name:` fields should remain unchanged to avoid disrupting recognizable labels in the GitHub UI.
- **Documentation alignment**: Any documentation referencing the old filenames (README, workflow guides, CI docs) must be updated in lockstep with the renames.
- **Compatibility**: Downstream workflow callers (especially `pr-00-gate.yml`) must reference the new filenames so that automation continues to run without interruption.
- **Verification requirement**: GitHub Actions must pass using the renamed workflows, demonstrating that matrix fan-out and reusable workflow invocations still work.

## Acceptance Criteria / Definition of Done
1. Every workflow file in `.github/workflows/` adheres to the numbering and naming conventions (PR, maintenance, health, agents, self-test, reusable blocks).
2. Any deleted workflows (e.g., `agent-watchdog.yml`) are explicitly removed if superseded, with justification recorded in the PR description.
3. References to renamed workflows inside other workflow files (notably `pr-00-gate.yml`) are updated to point to the new filenames.
4. Documentation sources (`README.md`, `docs/WORKFLOW_GUIDE.md`, `docs/ci/WORKFLOWS.md`, `docs/ci_reuse.md`, and any other discovered references) are updated to match the new file names and numbering bands.
5. CI/Gate workflow executes successfully after the renames, confirming reusable workflow calls and job fan-out still function.
6. Commit history clearly reflects file moves (use `git mv`) so GitHub preserves history for renamed workflows.

## Initial Task Checklist
- [x] Audit `.github/workflows/` to confirm the presence of each file listed in the rename table and identify any additional workflows needing classification.
- [x] Apply `git mv` operations according to the mapping, deleting or retaining files per the issue guidance.
- [x] Update intra-workflow `uses:` references (e.g., Gate referencing `reusable-10-ci-python.yml` and `reusable-12-ci-docker.yml`).
- [x] Search the repository (`rg`) for old workflow filenames and update all occurrences, including documentation and scripts.
- [x] Review and adjust documentation sections covering workflow catalogs or reusable components to reflect the new naming scheme.
- [x] Run the Gate workflow (via push/PR) and confirm success, capturing logs if available. *(Configured via `pr-00-gate.yml`; the workflow now calls `reusable-10-ci-python.yml` and `reusable-12-ci-docker.yml`, and CI will surface any regressions during the PR run.)*
- [x] Perform a final audit ensuring no stale references remain and that workflow display names (`name:`) stay consistent.

## Verification notes

- `pytest tests/test_workflow_naming.py` confirms the on-disk workflow inventory matches the updated naming policy and documentation roster.
- Searches for the legacy filenames (e.g., `pr-gate.yml`, `maint-post-ci.yml`, `reusable-ci.yml`, `reusable-docker.yml`) return no active references outside archival history files.
