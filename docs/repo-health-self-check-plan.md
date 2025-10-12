# Repo Health Workflow Remediation Plan

## Scope and Key Constraints
- Repair `.github/workflows/health-40-repo-selfcheck.yml` so the workflow validates, appears in the Actions list, and can be executed on demand.
- Keep the job read-only: rely on the built-in `contents`, `issues`, `pull-requests`, and `actions` read scopes and avoid introducing broader permissions or repository secrets.
- Preserve the existing triggers (`workflow_dispatch` plus an optional low-frequency schedule) while keeping total runtime under a minute on a GitHub-hosted runner.
- Produce a concise summary table in the step summary instead of failing the workflow for soft signals; only hard configuration or execution errors should mark the run as failed.

## Acceptance Criteria / Definition of Done
- The workflow definition passes GitHub validation, is no longer flagged as “Invalid workflow file,” and is visible in the Actions UI.
- A manual `workflow_dispatch` run completes on the default branch without permission errors and emits the repo health summary to `$GITHUB_STEP_SUMMARY`.
- (Optional) The scheduled trigger executes successfully and generates the same summary output.
- The permissions block requests only the minimum supported scopes required for read-only inspections.
- Supporting documentation in `docs/ci/WORKFLOWS.md` covers the workflow’s purpose, triggers, and permissions at a glance.

## Initial Task Checklist
- [x] Review the current workflow to catalog unsupported permission keys, disabled triggers, and health checks that should be preserved. (See updated `health-40-repo-selfcheck.yml` for the refined probes and summary step.)
- [x] Replace invalid permission entries with supported read-only scopes and confirm every step runs without elevated access. (Workflow now requests `contents`, `issues`, `pull-requests`, and `actions` read scopes only.)
- [x] Reconfirm the trigger configuration (`workflow_dispatch` and optional weekly cron) and ensure the job name/description make the workflow easy to discover. (Keeps the weekly cron at `20 6 * * 1` alongside manual dispatch.)
- [x] Add or refine steps that gather repository health signals and write a markdown table to `$GITHUB_STEP_SUMMARY` summarising the findings. (New Python summary step publishes the collected checks.)
- [ ] Smoke-test the workflow via `workflow_dispatch` on a branch copy, then on the default branch after merging, capturing screenshots or logs for validation notes. (Pending manual run once merged to default branch.)
- [x] Update `docs/ci/WORKFLOWS.md` with a one-line entry describing the workflow’s goal, triggers, and minimal permission set. (Catalog entry updated for Maint 35 self-check.)
