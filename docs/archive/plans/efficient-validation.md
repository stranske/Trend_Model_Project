# Efficient Codex Validation Workflow

## The Problem
The original validation process was too slow (1+ hour for basic fixes) because it:
- Ran all checks sequentially without auto-fixing
- Required manual investigation of each issue  
- No fast-feedback loop for common problems

## The Solution: Multi-Speed Validation

### 🚀 Ultra-Fast Fix (30 seconds)
```bash
# Auto-fix the most common issues
./scripts/fix_common_issues.sh
```
**Fixes**: formatting, type stubs, basic imports

### ⚡ Fast Check (1-2 minutes) 
```bash
# Quick validation - skips slow tests
./scripts/check_branch.sh --fast --fix
```
**Checks**: formatting, linting, type checking (auto-fixes what it can)

### 🔍 Full Validation (3-5 minutes)
```bash
# Complete validation with auto-fix
./scripts/check_branch.sh --fix
```
**Checks**: Everything including tests and coverage

### 🐛 Debug Mode
```bash
# Detailed output for troubleshooting
./scripts/check_branch.sh --verbose
```

## Efficient Workflow

### 1. After Codex Commits
```bash
# Start with ultra-fast fix
./scripts/fix_common_issues.sh

# Then do fast check
./scripts/check_branch.sh --fast
```
⏱️ **Total time: ~2 minutes**

### 2. Before Merging
```bash
# Full validation with auto-fix
./scripts/check_branch.sh --fix
```
⏱️ **Total time: ~5 minutes**

### 3. When Things Break
```bash
# Debug what's wrong
./scripts/check_branch.sh --verbose

# Fix specific issues manually
# Re-run fast check
./scripts/check_branch.sh --fast
```

## What Gets Auto-Fixed

✅ **Black formatting** - Automatic  
✅ **Missing type stubs** - Automatic  
✅ **Common import issues** - Automatic  
⚠️ **Line length** - Flagged for manual review  
❌ **Test coverage** - Manual (requires new tests)  
❌ **Complex linting** - Manual (requires code changes)

## Key Improvements

1. **Auto-fix capability** - No more manual formatting
2. **Fast mode** - Skip slow tests during development  
3. **Targeted fixes** - Fix common issues in 30 seconds
4. **Better feedback** - Clear status and next steps
5. **Time-boxed** - Each mode has a clear time expectation

## Usage Examples

```bash
# Quick daily validation
./scripts/fix_common_issues.sh && ./scripts/check_branch.sh --fast

# Pre-merge validation  
./scripts/check_branch.sh --fix

# Investigate failures
./scripts/check_branch.sh --verbose --fix
```

## Validation Example

```bash
$ ./scripts/check_branch.sh --fast --verbose
Running in fast mode - skip slow checks
Running in verbose mode - showing detailed output

=== Code Quality Validation ===
✓ Black formatting: PASSED
✓ Flake8 linting: PASSED
✓ MyPy type checking: PASSED

=== Validation Summary ===
🎉 All validations passed! Codex commits look good for merge
✓ Ready to merge or continue development
```

## Validation Example on Codex Branch

```bash
$ git checkout codex/update-demo-functionality-tests
# Fix the formatting issue in demo script
$ black scripts/run_multi_demo.py
# Verify that no further formatting changes are needed
$ black --check .
All done! ✨ 🍰 ✨
61 files left unchanged.

# Run tests to confirm functionality
$ pytest tests/ -q
122 passed in 2.47s
```

<!-- Ensure the validation scripts exist on the target branch -->
**Note:** Before running validation on `codex/update-demo-functionality-tests`, merge or cherry-pick the validation scripts into that branch so `scripts/check_branch.sh` is available.

This reduces validation time from **60+ minutes** to **2-5 minutes** for most cases!

## Running Validation Against Temporary Codex Branches

Since Codex often works on short-lived branches that may not include the validation scripts, you can run the scripts from your main or demo-pipeline branch without permanently merging them. For example:

```bash
# 1. Fetch the latest and check out the new Codex branch into a worktree
git fetch origin
git worktree add /tmp/codex-branch origin/<codex-branch>

# 2. Run validation from your stable branch's scripts directory
bash scripts/check_branch.sh --fast --verbose --branch /tmp/codex-branch
```

Here the `--branch` flag (or an environment variable) tells the validation script to point at a different working directory. If your scripts do not support `--branch`, you can temporarily set your working directory:

```bash
pushd /tmp/codex-branch
bash ../scripts/check_branch.sh --fast --verbose
popd
```

This approach lets you validate any Codex-generated branch— even if it does not include the validation scripts—without modifying that branch.
