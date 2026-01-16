# Exempt Template Sync PRs from Missing Issue Warning

## Why

Template sync PRs from stranske/Workflows are automated maintenance PRs that systematically sync workflow updates across consumer repos. They don't correspond to a tracked issue but are legitimate automated maintenance.

Currently, these PRs trigger a "missing issue" warning which creates noise and suggests action is required when none is needed.

## Current Behavior

PR #4383 (template sync) shows:
```
⚠️ Action Required: Unable to determine source issue for PR #4383.
```

## Expected Behavior

Template sync PRs should be exempt from the issue requirement, similar to how dependabot/renovate PRs are handled differently in agents-guard.js.

## Scope

Update PR meta workflow to recognize and exempt:
- PRs with `sync` + `automated` labels
- PRs with branch prefix `sync/`
- PRs with marker `<!-- meta:automated:template-sync -->`

## Tasks

- [ ] Update PR meta workflow (agents-pr-meta-v4.yml or reusable-20-pr-meta.yml) to detect template sync PRs
- [ ] Add exemption logic similar to existing bot exemptions in agents-guard.js
- [ ] Add test case for automated PR exemption
- [ ] Document the exemption pattern

## Acceptance Criteria

- Template sync PRs no longer show "missing issue" warning
- Regular PRs still show warning when appropriate
- Keepalive still skips correctly for template sync PRs
- Documentation updated

## Implementation Notes

Reference existing pattern in `.github/scripts/agents-guard.js`:
```javascript
const isAutomatedPR = normalizedAuthor && 
  (normalizedAuthor === 'dependabot[bot]' || normalizedAuthor === 'renovate[bot]');
```

Extend to also check:
```javascript
const isSyncPR = hasLabel('sync') && hasLabel('automated') || 
                 branchName.startsWith('sync/') ||
                 prBody.includes('<!-- meta:automated:template-sync -->');
```
