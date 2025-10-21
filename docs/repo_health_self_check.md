# Repository Health Workflow (health-41)

The `health-41-repo-health.yml` workflow runs a light-touch sweep of repository
hygiene each Monday at 07:15 UTC. It records a single Markdown report in the job
summary highlighting stale branches, open issues without an assignee, and the
default-branch protection state so the on-call maintainer can triage them
quickly. The report is generated directly with `actions/github-script`, so the
former `.github/scripts/repo_health_probe.py` wrapper has been removed and there
is no Python probe helper to maintain alongside the workflow.

## What it reports

| Signal | What it captures | Notes |
| ------ | ---------------- | ----- |
| Stale branches | Branches (excluding the default) whose latest commit is older than `REPO_HEALTH_STALE_BRANCH_DAYS`. | Sorted by oldest commit first and capped at 20 rows. |
| Unassigned issues | Open issues without assignees ordered by last update. | Includes links so triage is one click away. |
| Branch protection | Expected vs. enforced required status contexts on the default branch. | Fails when `Gate / gate` drops from branch protection and surfaces the current context list for audit. |

The summary also surfaces aggregate counts for each signal at the top of the
report.

## Trigger modes

- **Weekly cron** — Monday 07:15 UTC keeps the sweep in a quiet window.
- **`workflow_dispatch`** — Run manually after pruning branches or assigning
  issues to verify the report clears.

## Sample run summary

```
# Repository health weekly sweep
Generated on Mon, 03 Feb 2025 07:15:32 GMT

| Signal | Count |
| ------ | ----- |
| Stale branches (>30d) | 2 |
| Open issues without assignees | 3 |
```

Each section expands into a table with the specific branches or issues. When
more than 20 entries exist, the summary notes how many were omitted so you know
whether further cleanup is required. The branch-protection section lists the
expected context(s) alongside what GitHub currently enforces; drift triggers a
failure with language calling out the default branch so the responder knows
exactly where to restore the `Gate / gate` requirement.

## Outputs

- **Workflow summary** — The run log records the Markdown summary shown above,
  including the branch-protection comparison.
- **JSON artifact** — `repo-health-summary.json` captures the raw check payload
  (timestamps, branch status, label findings) for scripting or audit trails.
- **Optional PR checklist** — When the workflow is dispatched with
  `pull_request_number`, it posts or updates a comment containing the summary
  and a “How to fix” checklist. The checklist automatically marks the run green
  when only warnings remain.

## Tuning the sweep

The workflow reads the optional repository variable `REPO_HEALTH_STALE_BRANCH_DAYS`
to decide how old a branch must be before it is reported. Increase the value for
long-lived release branches or decrease it to surface drift sooner. The `MAX_TABLE_ROWS`
setting is hard-coded to 20 inside the workflow for concise reports.

After updating the threshold or restoring branch protection, dispatch the
workflow manually to confirm the new window behaves as expected and that the
`Gate / gate` requirement is back in place.
