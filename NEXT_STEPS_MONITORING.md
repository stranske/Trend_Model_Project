# Next Steps: Monitoring Task Appendix Fix

## ✅ Completed

1. **Root cause identified**: GitHub Actions secret scanner blocks `task_appendix` job output with long/repetitive content
2. **Immediate fix**: Deduplicated PR #4742 and issue #4735 (16→8 checkboxes)
3. **Systemic fix PR #1300**: Merged (but had critical issues in review comments)
4. **Review fixes PR #1306**: Created with 4 critical corrections
   - Write task_appendix directly in github-script (bypasses output blocking)
   - Stream file contents (avoid memory limits)
   - Updated to actions/download-artifact@v7
   - Mirrored to consumer template

## 🔄 In Progress

**Workflows PR #1306**: https://github.com/stranske/Workflows/pull/1306
- State: OPEN, mergeable
- CI: Gate running (Health 44 Gate Branch Protection: IN_PROGRESS)
- Waiting for: CI pass → merge

## 📋 Next Steps

### 1. After PR #1306 Merges

**Wait for template sync to Trend_Model_Project**:
```bash
# Monitor sync PR creation
gh pr list --search "sync workflow templates" --state open --limit 1

# Or manually trigger sync if needed
gh workflow run template-sync.yml --repo stranske/Workflows
```

### 2. Monitor First Keepalive Run with Fix

**Find a PR with keepalive activity**:
```bash
# List open agent PRs
gh pr list --label "agent:codex" --state open --limit 5

# Check recent keepalive runs
gh run list --workflow agents-keepalive-loop.yml --limit 10
```

**Check for artifact creation**:
```bash
# Get latest keepalive run
RUN_ID=$(gh run list --workflow agents-keepalive-loop.yml --limit 1 --json databaseId --jq '.[0].databaseId')

# Check if artifact was created
gh api repos/stranske/Trend_Model_Project/actions/runs/$RUN_ID/artifacts | \
  jq '.artifacts[] | select(.name | startswith("keepalive-task-appendix"))'
```

### 3. Verify Codex Receives Task List

**Download and inspect artifact**:
```bash
# Download artifact from run
gh run download $RUN_ID --name keepalive-task-appendix-PRNUM

# Check contents
cat task-appendix.txt
```

**Check assembled prompt logs**:
```bash
# View "Assemble prompt" step logs
gh run view $RUN_ID --log | grep -A 20 "Assemble prompt"

# Should show: "Streaming task appendix from artifact file"
# Or: "Reading task appendix from artifact file" (old message)
```

### 4. Confirm No Task Drift

**Monitor PR commits**:
```bash
# Check what files Codex is modifying
gh pr view PRNUM --json files --jq '.files[].path'

# Should match PR tasks (e.g., for turnover caps: simulation files, not cache config)
```

**Check task completion**:
```bash
# Count unchecked tasks over time
gh pr view PRNUM --json body --jq '.body' | grep -c "^- \[ \]"

# Should decrease as work progresses
```

## 🛡️ Validation Checklist

After PR #1306 merges and syncs:

- [ ] Template sync PR appears in Trend_Model_Project
- [ ] Next keepalive run creates `keepalive-task-appendix-*` artifact
- [ ] Artifact contains expected task list (not empty)
- [ ] Codex prompt assembly logs show artifact usage
- [ ] Codex works on correct tasks (matches PR body)
- [ ] Task checkboxes update as work completes
- [ ] No more "secret scanner blocking" warnings in logs

## 📊 Known Issues Resolved

| Issue | Old Behavior | New Behavior |
|-------|--------------|--------------|
| Secret scanner blocking | Output censored, artifact empty | Direct file write, bypasses output |
| Memory limits | `appendix_content=$(cat file)` | Stream with `cat file >> output` |
| Template drift | Consumer template outdated | Mirrored changes included |
| Action versions | Using v6 | Updated to v7 |

## 🔗 Related Links

- **Blocking issue**: stranske/Trend_Model_Project#4735
- **Deduplicated content**: stranske/Trend_Model_Project#4742 (closed/merged?)
- **Original fix**: stranske/Workflows#1300 (merged with issues)
- **Review fixes**: stranske/Workflows#1306 (open, awaiting merge)
- **Evidence**: https://github.com/stranske/Trend_Model_Project/actions/runs/21777658244 (secret scanner warning)

## 💡 If Problems Persist

**Symptom**: Artifact still empty
- Check: Was PR #1306 actually merged?
- Check: Did template sync complete?
- Check: Is consumer repo using latest workflow version?

**Symptom**: Still seeing task drift
- Check: PR body deduplicated? (should be 8 items, not 16)
- Check: Does artifact contain correct tasks?
- Check: Are there other duplicate issues (#4735 pattern)?

**Symptom**: Agent working on wrong code
- Check: Assembled prompt includes "## Run context" section?
- Check: Task list matches PR body exactly?
- Compare: Artifact content vs what Codex received
