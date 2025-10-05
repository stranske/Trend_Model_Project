<!-- bootstrap for codex on issue #1205 -->

# TTL enforcement for stale pull requests (Issue #1205)

## Objective
Establish automated lifecycle management for inactive pull requests so queues stay actionable and contributors receive timely nudges before closure.

## High-level requirements
- Detect pull requests with no activity (comments, reviews, pushes) for **N** days, add a `stale` label, and post a reminder comment that references the TTL window and opt-out instructions.
- If inactivity continues for **M** days after the warning (M > N), post a closure notice and automatically close the pull request while leaving the branch intact.
- Ensure the automation runs on a scheduled cadence (e.g., daily) and can be manually dispatched for testing.
- Avoid touching PRs that are explicitly exempt (labels such as `keep-open`, ongoing reviews, draft state, or linked deployment blockers).

## Implementation outline
1. **Workflow scaffolding**
   - Add a new scheduled GitHub Actions workflow (likely under `.github/workflows/stale-pr-ttl.yml`).
   - Use `actions/github-script` or a small Python script to query open PRs via the REST API filtered by updated timestamp.
   - Parameterize TTL values via workflow `env` or repository variables so operations can adjust without code changes.

2. **Stale detection + reminder path**
   - When a PR surpasses the first inactivity threshold (N days) and lacks exempt labels/states, apply `status:stale` (or repo-standard equivalent) and post a comment outlining the impending closure timeline.
   - Record the reminder timestamp using either the label's creation timestamp (available via the GitHub API) or a dedicated comment with machine-readable metadata, so the follow-up phase can be calculated deterministically.

3. **Auto-close path**
   - For PRs already labeled stale, compare the reminder timestamp to `M`. If exceeded and no new commits/comments, reviews, or pushes have occurred *since the reminder timestamp*, post a closure message and call the REST API to close the PR.
   - Remove/replace labels to reflect the closed state (e.g., `status:closed-by-ttl`) for analytics and to prevent further processing.

4. **Safety rails & observability**
   - Dry-run mode flag for initial rollout to print intended actions without mutating PRs.
   - Structured summary in the workflow run (counts of nudged/closed/exempt PRs).
   - Metrics-friendly JSON artifact capturing processed PR numbers and decisions for later auditing.

5. **Testing & validation**
   - Unit-test any helper scripts (if Python) using the GitHub API fixtures.
   - Add integration coverage by mocking API calls through recorded responses (e.g., `responses` library) to ensure label/comment logic behaves with multiple edge cases.
   - Exercise the workflow via `workflow_dispatch` against a sandbox branch to confirm permissions and comment formatting.

## Open questions / follow-ups
- Confirm the canonical label names for "stale" and "keep open" to match repository conventions.
- Determine whether draft PRs should be exempt by default or receive a separate TTL clock.
- Decide on localization/templating strategy for reminder + closure comments (single template vs. separate markdown partials).
- Align with existing cleanup workflows (e.g., `maint-32-autofix.yml` and `maint-33-check-failure-tracker.yml`) to avoid overlapping responsibilities.

## Definition of done
- Scheduled workflow merged on `phase-2-dev`, documented in `docs/ops/codex-bootstrap-facts.md` (automation inventory) and `docs/agent-automation.md` if necessary.
- Successful dry-run showcasing accurate categorization, followed by live run closing at least one synthetic stale PR (or verified via mocked scenario).
- Playbook entry (in `docs/agent_codex_troubleshooting.md` or new page) describing how to override/opt-out for special cases.
