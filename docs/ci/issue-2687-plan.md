# Issue #2687 Gate Docs-Only Fast Path – Planning Notes

## Scope and Key Constraints
- **Docs-only detection alignment**: Limit code changes to Gate workflow logic and adjacent helpers that decide whether a pull request is docs-only; avoid introducing new workflows or altering unrelated CI pipelines.
- **Workflow pruning**: Remove the legacy `.github/workflows/pr-14-docs-only.yml` file (or confirm its absence) without disturbing other required status checks or workflow naming conventions.
- **Gate signal continuity**: Preserve Gate’s public contract—still expose a single required check whose status remains green for docs-only PRs while skipping heavyweight jobs.
- **Documentation clarity**: Update existing workflow documentation (e.g., `docs/ci/WORKFLOW_SYSTEM.md`) instead of creating brand-new guides, keeping language consistent with established CI docs style.
- **Backward compatibility**: Ensure non-docs-only PRs still execute all required Gate jobs; changes must not degrade coverage detection or job fan-out for code-touching diffs.

## Acceptance Criteria / Definition of Done
1. `.github/workflows/pr-14-docs-only.yml` is removed from the repository (or verified absent) and protected by a regression guard so the file cannot reappear.
2. Gate workflow detects docs-only changes, surfaces a single skip notice via logs/step summary, and marks the overall required status as successful without running heavy jobs.
3. Non-docs-only changes still trigger the full Gate job matrix, with no regressions in required check names or status reporting.
4. `docs/ci/WORKFLOW_SYSTEM.md` (or equivalent primary CI documentation) is updated to describe the consolidated docs-only fast path behavior.
5. Automated tests cover both the absence of the PR-14 workflow file and the Gate docs-only branching logic (unit or integration as appropriate).
6. CI runs for the updated branch show Gate succeeding on a docs-only change and executing normally on a non-docs-only change (evidence recorded in PR description or summary).

## Initial Task Checklist
- [x] Audit `.github/workflows/` to confirm the legacy `pr-14-docs-only.yml` file status and identify any lingering references.
- [x] Extend Gate workflow (or its helpers) to short-circuit heavy jobs when `docs_only` is detected while still surfacing a single skip notice in logs/summary.
- [x] Add or update automated tests that enforce the absence of the deprecated workflow file and validate docs-only detection logic.
- [x] Refresh `docs/ci/WORKFLOW_SYSTEM.md` to document the consolidated Gate docs-only fast path and removal of PR-14 workflow.
- [x] Run targeted CI/tests (e.g., Gate workflow via PR, focused pytest modules) to demonstrate docs-only fast pass and regular execution paths.
- [x] Capture evidence (links/screenshots/log references) of Gate behavior for both docs-only and non-docs-only scenarios and summarize in the PR description.
