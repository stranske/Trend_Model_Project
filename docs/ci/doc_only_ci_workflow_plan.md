# Doc-Only CI Workflow Planning Notes

## Scope and Key Constraints
- Implement a GitHub Actions workflow that runs only for pull requests modifying documentation-specific paths (`**/*.md`, `docs/**`, `assets/**`).
- The workflow must detect doc-only diffs by combining `paths` filters with a guard that prevents execution when other file types are present.
- Use `actions/github-script@v7` within the job to author a single pull request comment summarizing the doc-only detection result.
- Keep the workflow lightweight: avoid checkout or dependency installs, and complete within seconds to act as a quick signal.
- Ensure the workflow is non-disruptive to existing CI by using a distinct job name (`docs-only`) and by exiting early when the change set includes non-doc files.

## Acceptance Criteria / Definition of Done
- A GitHub Actions workflow file exists under `.github/workflows/` defining the `docs-only` job.
- The job triggers for pull request events and is limited by documentation path filters. A follow-on detection step keeps the
  commenting job dormant when any non-doc files are present in the diff.
- When executed, the workflow posts exactly one comment on the pull request containing the message: `Doc‑only change detected; Gate will still run as the required check.`
- The workflow avoids posting duplicate comments if rerun on the same PR by updating an existing comment or ensuring idempotent logic.
- The workflow has been linted/validated (e.g., via `act -n` or GitHub Actions workflow syntax check) to confirm there are no YAML or logic errors.
- When running the broader `scripts/workflow_lint.sh` helper, expect unrelated legacy findings; targeted invocations such as
  `./.cache/actionlint/actionlint .github/workflows/pr-14-docs-only.yml` keep validation noise-free for this workflow.
- Documentation describing the workflow purpose and limitations is added to the repository.

## Initial Task Checklist
- [x] Draft workflow YAML (`.github/workflows/pr-14-docs-only.yml`) with appropriate event triggers and path filters.
- [x] Implement `actions/github-script@v7` steps that locate any previous automation comment and create or update the doc-only notification.
- [x] Add inline comments explaining how the doc-only detection works and how to modify paths if documentation scope evolves.
- [x] Document the workflow in `docs/ci/` (summary, triggers, comment behavior, and maintenance tips).
- [x] Run a workflow linter (`scripts/workflow_lint.sh`) to validate syntax and guard conditions.
- [x] Request review/approval and confirm comment appears as expected on a doc-only PR.
