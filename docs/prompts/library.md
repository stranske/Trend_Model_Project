# Prompt Library

Quick-reference prompts for common tasks. Copy and paste as needed.

---

## Verification & Preflight

Prompts that ensure factual grounding before analysis.

1. **PR Preflight Check**
   Run before any PR evaluation or analysis.
   ```
   Run ./scripts/preflight-pr.sh <PR_NUMBER> and show me the output before any analysis.
   ```

2. **Issue State Check**
   Verify issue state before making claims.
   ```
   gh issue view <ISSUE_NUMBER> --json title,state,labels,assignees,body
   ```

3. **Workflow Run Status**
   Check actual workflow status on a branch.
   ```
   gh run list --branch <BRANCH> --limit 10
   ```

4. **Compare Branch to Remote**
   Verify local vs remote state before pushing.
   ```
   git fetch origin && git log HEAD..origin/<BRANCH> --oneline && git log origin/<BRANCH>..HEAD --oneline
   ```

---

## Agent Instructions

Standard prompts for instructing coding agents.

1. **Standard Keepalive Instruction**
   The canonical instruction format for Codex agent rounds. Copy exact text from `.github/templates/keepalive-instruction.md`.
   ```
   @codex Your objective is to satisfy the **Acceptance Criteria** by completing each **Task** within the defined **Scope**.

   **This round you MUST:**
   1. Implement actual code or test changes that advance at least one incomplete task toward acceptance.
   2. Commit meaningful source code (.py, .yml, .js, etc.)—not just status/docs updates.
   3. Mark a task checkbox complete ONLY after verifying the implementation works.
   4. **POST A REPLY COMMENT** with completed checkboxes using the **EXACT TEXT** from the lists below.

   **CRITICAL - Checkbox Format:**
   When posting your reply, copy the **exact checkbox text** from the Tasks and Acceptance Criteria sections below. Do NOT paraphrase or summarize. The automation matches text exactly.

   Example format for your reply:
   ## Completed this round
   ### Tasks
   - [x] Add an environment flag (e.g., `TREND_PIPELINE_PROXY_SIMPLE`) that, when set, skips GC scanning...
   ### Acceptance Criteria  
   - [x] Setting the new env flag yields a direct-import proxy path; default continues to honour GC-scanned modules.

   **DO NOT:**
   - Commit only status files, markdown summaries, or documentation when tasks require code.
   - Re-post checklists without making implementation progress.
   - Close the round without source-code changes when acceptance criteria require them.
   - Paraphrase or shorten checkbox text—copy it exactly for tracking to work.

   Review the Scope/Tasks/Acceptance below, identify the next incomplete task that requires code, implement it, then **post a reply comment** with the completed items using their **exact original text**.
   ```

---

## Issue & Acceptance Evaluation

Prompts for evaluating issue completion status.

1. **Evaluate Issue Acceptance Criteria**
   Determine if an issue can be closed by reviewing actual code and outcomes.
   ```
   Evaluate Issue <ISSUE_NUMBER> at https://github.com/stranske/Trend_Model_Project/issues/<ISSUE_NUMBER> to determine whether all acceptance criteria have been met and the issue can be closed, or whether work remains.

   Requirements:
   - Run ./scripts/preflight-pr.sh on any linked PRs first
   - Review actual code changes in the remote repo, not summaries
   - Verify each acceptance criterion against committed code and test results
   - Check that tests exist and pass for claimed functionality
   - Report: which criteria are MET (with evidence), which are NOT MET (with gaps)
   ```

2. **PR Acceptance Check**
   Verify a PR satisfies its linked issue's acceptance criteria.
   ```
   For PR #<PR_NUMBER>, verify each acceptance criterion from the linked issue is satisfied.
   Run ./scripts/preflight-pr.sh <PR_NUMBER> first, then check the actual diff and test results.
   ```

---

## Code Quality & Validation

Prompts for ensuring code meets standards.

1. **Quick Validation (Development)**
   Fast check during active development.
   ```
   ./scripts/dev_check.sh --changed --fix
   ```

2. **Pre-Commit Validation**
   Run before committing changes.
   ```
   ./scripts/validate_fast.sh --fix
   ```

3. **Comprehensive Validation**
   Full validation before merge.
   ```
   ./scripts/check_branch.sh --fast --fix
   ```

4. **Run Specific Test File**
   Test a specific file after changes.
   ```
   PYTHONPATH="./src" pytest <test_file_path> -v
   ```

---

## Git Operations

Prompts for git workflow tasks.

1. **Check Changed Files**
   See what's modified before committing.
   ```
   git status && git diff --stat
   ```

2. **Interactive Rebase**
   Clean up commits before push.
   ```
   git rebase -i HEAD~<N>
   ```

3. **Check PR Status**
   Verify PR checks and mergeability.
   ```
   gh pr checks <PR_NUMBER>
   ```

4. **Sync with Base Branch**
   Update branch with latest from base.
   ```
   git fetch origin && git rebase origin/<BASE_BRANCH>
   ```

---

## Agent & Keepalive Operations

Prompts for working with the agent automation system.

1. **Check Keepalive State**
   Verify keepalive labels and comments on a PR.
   ```
   ./scripts/preflight-pr.sh <PR_NUMBER>
   ```

2. **List Agent Workflow Runs**
   Check agent-related workflow activity.
   ```
   gh run list --workflow=agents-70-orchestrator.yml --limit 5
   gh run list --workflow=agents-pr-meta-v4.yml --limit 5
   ```

3. **Check Workflow File Version**
   Verify which workflow file version is running.
   ```
   gh api repos/stranske/Trend_Model_Project/actions/runs/<RUN_ID> --jq '{path, workflow_name: .name, conclusion}'
   ```

4. **View Workflow Run Logs**
   Get logs from a specific run.
   ```
   gh run view <RUN_ID> --log
   ```

---

## Debugging & Investigation

Prompts for troubleshooting issues.

1. **Check CI Failure Details**
   Get failed job output from a run.
   ```
   gh run view <RUN_ID> --log-failed
   ```

2. **Search for Error in Logs**
   Find specific error in workflow logs.
   ```
   gh run view <RUN_ID> --log 2>&1 | grep -A10 "<ERROR_PATTERN>"
   ```

3. **Check File History**
   See recent changes to a file.
   ```
   git log --oneline -10 <FILE_PATH>
   ```

4. **Diff Against Base Branch**
   See all changes in current branch.
   ```
   git diff origin/<BASE_BRANCH>...HEAD --stat
   ```

---

## Session Management

Prompts for managing work sessions.

1. **Session Status Check**
   Quick overview of current state.
   ```
   git status && git log --oneline -3 && gh pr list --author @me
   ```

2. **Todo List Check**
   Review current task list (for Copilot).
   ```
   Read the current todo list.
   ```

3. **Stash Work in Progress**
   Save uncommitted work temporarily.
   ```
   git stash push -m "<description>"
   ```

4. **Resume Stashed Work**
   Restore previously stashed changes.
   ```
   git stash list && git stash pop
   ```
