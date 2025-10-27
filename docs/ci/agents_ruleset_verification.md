# Agents ruleset validation log

## 2025-09-05 — Ruleset health check

- Queried the repository rulesets via the public REST API to confirm the guard
  that blocks deletions and renames of the Agents workflows.
- Identified three rulesets (`Protect Phase 1`, `Protect Phase 2`, and `Tests
  Pass to Merge`). Only `Tests Pass to Merge` targets the default branch, but it
  is currently disabled and does not expose file-specific protections.
- Follow-up required: re-enable `Tests Pass to Merge` (ruleset ID `7880490`) and
  populate its `restrict_file_updates` rule with the three workflow paths so the
  enforcement matches policy.

### Commands executed

```bash
# List available repository rulesets and surface enforcement state
curl -s "https://api.github.com/repos/stranske/Trend_Model_Project/rulesets?per_page=100&includes_parents=true" \
  | jq '.[] | {id, name, enforcement}'

# Inspect default-branch ruleset details
curl -s https://api.github.com/repos/stranske/Trend_Model_Project/rulesets/7880490 \
  | jq '{name, enforcement, scope: .conditions.ref_name.include,
         rules: [.rules[] | {type, parameters}]}'
```

### Key command output

```json
{
  "id": 7880490,
  "name": "Tests Pass to Merge",
  "enforcement": "disabled"
}

{
  "name": "Tests Pass to Merge",
  "enforcement": "disabled",
  "scope": [
    "~DEFAULT_BRANCH",
    "~ALL"
  ],
  "rules": [
    { "type": "deletion", "parameters": null },
    { "type": "non_fast_forward", "parameters": null },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": false,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          { "context": "CI / test (3.11)" },
          { "context": "Docker / build" }
        ]
      }
    }
  ]
}
```

The commands returned the JSON snippets captured above, showing the ruleset is
disabled and the `rules` array lacks any `restrict_file_updates` entries. The
other two rulesets apply to the phase branches exclusively.

### Next steps
- Coordinate with a repository admin to enable `Tests Pass to Merge` and add a
  `restrict_file_updates` rule that lists:
  - `.github/workflows/agents-63-issue-intake.yml`
  - `.github/workflows/agents-70-orchestrator.yml`
- Capture a screenshot of the GitHub UI after the update and link it here for
  auditability once the rule is active.
