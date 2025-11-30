#!/bin/bash
# Preflight check for PR evaluation - run this BEFORE making any claims about PR state
# Usage: ./scripts/preflight-pr.sh <PR_NUMBER>

set -e

PR="${1:?Usage: preflight-pr.sh <PR_NUMBER>}"
REPO="${2:-stranske/Trend_Model_Project}"

echo "=== PREFLIGHT: PR #$PR ==="
echo ""

echo "--- PR State & Labels ---"
gh pr view "$PR" --repo "$REPO" --json number,title,state,labels,author,createdAt,headRefName \
  --jq '{number, title, state, labels: [.labels[].name], author: .author.login, createdAt, branch: .headRefName}'
echo ""

echo "--- Comments (last 10) ---"
gh api "repos/$REPO/issues/$PR/comments?per_page=10" \
  --jq '.[] | {id, author: .user.login, created_at, body_preview: .body[0:200]}'
echo ""

echo "--- Recent Workflow Runs ---"
BRANCH=$(gh pr view "$PR" --repo "$REPO" --json headRefName -q .headRefName)
gh run list --repo "$REPO" --branch "$BRANCH" --limit 10 2>/dev/null || echo "(no runs found)"
echo ""

echo "--- Workflow Run Paths ---"
gh api "repos/$REPO/actions/runs?branch=$BRANCH&per_page=20" \
  --jq '[.workflow_runs[] | {name, path, conclusion}] | group_by(.path) | .[] | {path: .[0].path, count: length, conclusions: ([.[] | .conclusion] | unique)}' 2>/dev/null || echo "(no runs found)"
echo ""

echo "=== END PREFLIGHT ==="
