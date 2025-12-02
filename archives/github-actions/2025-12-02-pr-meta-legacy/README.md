# Archived: Legacy agents-pr-meta Workflows

**Archived:** 2025-12-02  
**Reason:** Replaced by `agents-pr-meta-v4.yml`

## Files

| File | Description |
|------|-------------|
| `agents-pr-meta.yml` | Original PR meta manager. Workflow ID became corrupted in GitHub's registry. Had already disabled triggers but kept for agents:guard reference. |
| `agents-pr-meta-v2.yml` | Copy created to work around corrupted workflow ID. Active triggers were still running and failing. |
| `agents-pr-meta-v3.yml` | Stub workflow used for testing trigger isolation. Never fully implemented. |

## Replacement

All functionality is now handled by `.github/workflows/agents-pr-meta-v4.yml`, which:
- Uses external scripts to stay under GitHub's workflow parser limits
- References `agents_pr_meta_update_body.js` for PR body updates
- Has identical triggers and permissions to v2
- Is actively maintained and passing CI

## Migration Notes

- v4 was introduced to resolve GitHub workflow file size issues
- The `agents_pr_meta_update_body.js` script contains the core logic previously embedded in the workflow YAML
- No action needed for existing PRs—v4 handles all PR metadata management
