## Problem

PR #1300 was merged with several unaddressed inline code review comments from agents. These comments identified critical issues that would prevent the fix from working correctly:

1. **CRITICAL**: Task appendix written from blocked output (defeats the purpose)
2. Memory issue: Reading large files into shell variables
3. Template drift: Changes not mirrored to consumer template  
4. Version inconsistency: Using v6 instead of v7

## Review Comments

### Comment 1: Critical - File Written from Blocked Output (Line 329)

**Issue**: The artifact is being written from `steps.evaluate.outputs.task_appendix`, but when GitHub's secret scanner blocks the output, this will be empty, so the uploaded artifact will also be empty and the fix won't work.

**Fix**: Write `result.taskAppendix` directly to file inside the `actions/github-script` step using `fs.writeFileSync` before it ever goes through the output mechanism.

### Comment 2: Memory Limits (Line 503)

**Issue**: `appendix_content=$(cat …)` reads the entire appendix into a shell variable. For very large/repetitive task lists, this can become memory-heavy and may hit shell/process limits.

**Fix**: Stream file contents directly into `$output` using `cat file >> output` instead of loading into a variable. Only use variable for small input parameter fallback.

### Comment 3: Template Drift (Line 337)

**Issue**: Workflow is consumer-synced per `.github/sync-manifest.yml`, but changes weren't mirrored to `templates/consumer-repo/.github/workflows/agents-keepalive-loop.yml`.

**Fix**: Applied identical changes to consumer template to prevent template drift.

### Comment 4: Version Consistency (Line 446)

**Issue**: New step uses `actions/download-artifact@v6`, but repo has moved to `@v7`.

**Fix**: Updated to `@v7` for consistency and latest fixes.

## Changes

### `.github/workflows/agents-keepalive-loop.yml`
- Move file writing into github-script step using `fs.writeFileSync`
- Write before `core.setOutput` to ensure content bypasses secret scanner
- Added verification step to confirm file exists
- Updated to `actions/upload-artifact@v7`

### `.github/workflows/reusable-codex-run.yml`
- Stream file contents directly: `cat file >> output`
- Removed intermediate variable for file content
- Updated to `actions/download-artifact@v7`

### `templates/consumer-repo/.github/workflows/agents-keepalive-loop.yml`
- Applied full fix (was missing from PR #1300)
- Prevents template drift
- Ensures consumer repos get the working fix

## Testing

- All fixes address root causes identified in code review
- Backward compatible (input parameter fallback preserved)
- Template now matches source workflow

## Related

- Original PR: #1300
- Blocked PR: stranske/Trend_Model_Project#4742
