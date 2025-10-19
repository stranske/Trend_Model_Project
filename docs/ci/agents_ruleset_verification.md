# Agents ruleset validation — 2025-09-04

## Summary
- Queried the repository rulesets via the public REST API to confirm the guard
  that blocks deletions and renames of the Agents workflows.
- Identified three rulesets (`Protect Phase 1`, `Protect Phase 2`, and `Tests
  Pass to Merge`). Only `Tests Pass to Merge` targets the default branch, but it
  is currently disabled and does not expose file-specific protections.
- Follow-up required: re-enable `Tests Pass to Merge` (ruleset ID `7880490`) and
  populate its `restrict_file_updates` rule with the three workflow paths so the
  enforcement matches policy.

## Commands executed

```bash
# List available repository rulesets
curl -s "https://api.github.com/repos/stranske/Trend_Model_Project/rulesets?per_page=100&includes_parents=true" \
  | jq '.[].name'

# Inspect default-branch ruleset details
curl -s https://api.github.com/repos/stranske/Trend_Model_Project/rulesets/7880490 \
  | jq '{name, enforcement, scope: .conditions.ref_name.include, rules}'
```

Both commands returned the JSON snippets captured in CI logs, showing
`"enforcement": "disabled"` for `Tests Pass to Merge` and a `rules` array without
any `restrict_file_updates` entries. The other two rulesets apply to the phase
branches exclusively.

## Next steps
- Coordinate with a repository admin to enable `Tests Pass to Merge` and add a
  `restrict_file_updates` rule that lists:
  - `.github/workflows/agents-63-chatgpt-issue-sync.yml`
  - `.github/workflows/agents-63-codex-issue-bridge.yml`
  - `.github/workflows/agents-70-orchestrator.yml`
- Capture a screenshot of the GitHub UI after the update and link it here for
  auditability once the rule is active.
