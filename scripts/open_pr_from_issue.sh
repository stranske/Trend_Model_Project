#!/usr/bin/env bash
# Open a PR as the current authenticated GitHub CLI user using an existing branch prepared by the workflow.
# Usage:
#   scripts/open_pr_from_issue.sh <issue_number> <branch_name> [base_branch]
#
# Requires: gh (GitHub CLI) authenticated as the human author.

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh (GitHub CLI) not found on PATH" >&2
  exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 <issue_number> <branch_name> [base_branch]" >&2
  exit 1
fi

ISSUE="$1"
BRANCH="$2"
BASE="${3:-}"

# Determine owner/repo and default base if not provided
OWNER_REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
if [ -z "${BASE}" ]; then
  BASE=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name)
fi

# Fetch issue details
TITLE=$(gh issue view "$ISSUE" --json title -q .title || echo "")
BODY=$(gh issue view "$ISSUE" --json body -q .body || echo "")
ISSUE_URL="https://github.com/${OWNER_REPO}/issues/${ISSUE}"

# Build PR body in a temp file (quote the issue body for readability)
TMPFILE=$(mktemp)
{
  echo "### Source Issue #${ISSUE}: ${TITLE}"
  echo
  echo "Source: ${ISSUE_URL}"
  echo
  # Quote each line of the issue body
  if [ -n "$BODY" ]; then
    echo "$BODY" | awk '{ print "> "$0 }'
    echo
  fi
  echo "—"
  echo "Next steps for the PR author:"
  echo "- Comment \`@codex start\` so Codex drafts the plan."
  echo "- After Codex replies with the checklist, post the execution command below to begin delivery and enable keepalive."
  echo
  echo "Execution command (copy into a standalone PR comment):"
  echo '\`\`\`markdown'
  echo '@codex plan-and-execute'
  echo
  echo 'Codex, reuse the scope, acceptance criteria, and task list from the source issue.'
  echo 'Post those sections on this PR using markdown checklists (- [ ]) so the keepalive workflow continues nudging until everything is complete.'
  echo 'Work through the tasks, checking them off only after each acceptance criterion is satisfied.'
  echo '\`\`\`'
} >"$TMPFILE"

# Create the PR as the current user
gh pr create \
  --head "$BRANCH" \
  --base "$BASE" \
  --title "Codex bootstrap for #${ISSUE}" \
  --body-file "$TMPFILE"

rm -f "$TMPFILE"

echo "Opened PR from ${BRANCH} into ${BASE} for issue #${ISSUE} as $(gh auth status -h github.com --show-token=false 2>/dev/null | sed -n 's/.*Logged in to github.com as \([^ ]*\).*/\1/p')"
