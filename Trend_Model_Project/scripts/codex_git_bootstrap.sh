#!/bin/bash
# codex_git_bootstrap.sh

set -e

# Replace with your actual branch name
BRANCH_NAME="main"
REMOTE_NAME="origin"

echo "🧹 Checking for uncommitted changes..."
if ! git diff-index --quiet HEAD --; then
  echo "❗ Uncommitted changes exist. Stash or commit them before proceeding."
  exit 1
fi

echo "🔄 Fetching latest from $REMOTE_NAME..."
git fetch "$REMOTE_NAME"

echo "🚀 Checking out $BRANCH_NAME..."
git checkout "$BRANCH_NAME" || git checkout -b "$BRANCH_NAME" "$REMOTE_NAME/$BRANCH_NAME"

echo "📥 Pulling latest changes..."
git pull "$REMOTE_NAME" "$BRANCH_NAME"

echo "✅ You are now on: $(git branch --show-current)"
