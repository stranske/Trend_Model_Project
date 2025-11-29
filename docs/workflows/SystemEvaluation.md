# Workflow System Evaluation

## Executive Summary

This document provides a technical evaluation of the current GitHub Actions workflow system, focusing on reported issues with API rate limits and the "keepalive" functionality.

**Key Findings:**
1.  **API Rate Limits**: The system uses aggressive polling loops and frequent API calls (e.g., creating PRs for simple branch syncs), which likely exhaust the GitHub API quota, especially during active development.
2.  **Keepalive Branch Sync**: The current mechanism for syncing branches is overly complex and fragile. It creates a temporary Pull Request to merge the base branch into the feature branch, which is prone to failure ("tripping") and triggers unnecessary CI runs.
3.  **Complexity**: The workflows rely heavily on large, inline JavaScript blocks within `actions/github-script`, making maintenance and debugging difficult.

---

## 1. API Rate Limits

The "API limits" issue is likely caused by a combination of the following factors:

### A. Aggressive Polling in Orchestrator
The `agents-70-orchestrator.yml` workflow contains a polling loop that runs for up to 3 minutes, checking for reactions and workflow runs every 5-15 seconds.

*   **Location**: `agents-70-orchestrator.yml` (Job: `keepalive-instruction`, Step: `Ack keepalive instruction`)
*   **Behavior**:
    ```javascript
    while (Date.now() < deadline) {
      // 1. List reactions (API call)
      const reactions = await github.paginate(github.rest.reactions.listForIssueComment, ...);
      
      // 2. List workflow runs (API call)
      if (Date.now() - lastRunCheck >= runPollInterval) {
         const response = await github.rest.actions.listWorkflowRuns(...);
      }
      
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
    ```
*   **Impact**: A single run can generate dozens of API calls. If multiple orchestrators run in parallel (or overlap), this consumes quota rapidly.

### B. "Heavy" Branch Sync
The `agents-keepalive-branch-sync.yml` workflow uses `peter-evans/create-pull-request` to create a PR just to merge the base branch into the feature branch.
*   **Impact**: Creating a PR involves multiple API calls (create branch, create PR, add labels, etc.).
*   **Side Effect**: Opening a PR triggers *all* `on: pull_request` workflows (linting, testing, etc.), doubling the CI load and API usage for what should be a simple git operation.

### C. Event-Heavy Triggers
`agents-pr-meta.yml` runs on `issue_comment` (created), `pull_request` (opened, synchronize, reopened, edited).
*   **Impact**: Every comment and every commit push triggers this heavy workflow, which performs multiple API checks (gates, labels, etc.).

---

## 2. Keepalive Branch Sync Failures

The user reported that "keepalive functionality never quite iterates until all tasks are completed due to tripping on updating branches."

### Analysis of `agents-keepalive-branch-sync.yml`
The current workflow performs the following dance:
1.  Checkout PR head.
2.  Create a temporary branch `sync/codex-{trace}`.
3.  Fetch and merge `origin/base_ref` into this temporary branch.
4.  Create a PR from `sync/codex-{trace}` back to the original PR branch.
5.  Auto-merge this new PR.

### Failure Modes
1.  **Merge Conflicts**: The step `git merge --no-commit --no-ff origin/"$BASE_REF"` will fail immediately if there are conflicts between the feature branch and the base branch. This causes the workflow to exit with an error, stopping the keepalive loop.
2.  **Race Conditions**: If the feature branch is updated (e.g., by the user or another agent) while this workflow is running, the PR creation or merge might fail.
3.  **Complexity**: Using a PR to perform a merge is an unnecessary abstraction that adds points of failure.

### Recommendation
Simplify the sync process to a direct git merge and push operation.
*   **Proposed Logic**:
    ```bash
    git checkout <feature-branch>
    git pull origin <feature-branch>
    git fetch origin <base-branch>
    git merge origin/<base-branch>
    git push origin <feature-branch>
    ```
*   **Conflict Handling**: If the merge fails due to conflicts, the workflow should catch the error and post a comment on the PR alerting the user that manual intervention is required, rather than just failing silently or crashing the loop.

---

## 3. Recommendations

### Immediate Fixes
1.  **Refactor Branch Sync**: Replace the PR-based sync in `agents-keepalive-branch-sync.yml` with a direct git merge/push script. This reduces API usage and complexity.
2.  **Reduce Polling**: Increase the polling interval in `agents-70-orchestrator.yml` from 5s to 30s, or remove the polling entirely in favor of event-driven triggers (e.g., `repository_dispatch`).

### Long-Term Improvements
1.  **Externalize Scripts**: Move the large inline JavaScript blocks from YAML files into dedicated `.js` files in `.github/scripts/`. This allows for unit testing and better version control.
2.  **Optimize Triggers**: Review `agents-pr-meta.yml` to ensure it only runs when necessary (e.g., ignore comments from bots to prevent loops).
3.  **Caching**: Implement caching for API results where possible (e.g., if checking the same PR status multiple times).

## Proposed Action Plan

1.  **Modify `agents-keepalive-branch-sync.yml`**:
    *   Remove `peter-evans/create-pull-request`.
    *   Implement direct `git merge` and `git push`.
    *   Add error handling for merge conflicts.

2.  **Modify `agents-70-orchestrator.yml`**:
    *   Relax the polling loop (increase `pollDelay`).

3.  **Documentation**:
    *   Update `Agents.md` to reflect the simplified sync process.
