# Pre-Workflows Migration Archive

**Date**: 2025-12-30
**Reason**: Transition to stranske/Workflows consumer pattern

## Contents

These 35 workflows were archived during the transition from local implementations
to the centralized stranske/Workflows consumer pattern.

### Archived Workflow Categories

| Category | Count | Files |
|----------|-------|-------|
| Reusables | 5 | `reusable-10-ci-python.yml`, `reusable-12-ci-docker.yml`, `reusable-16-agents.yml`, `reusable-18-autofix.yml`, `reusable-agents-issue-bridge.yml` |
| Agents | 12 | `agents-63-*`, `agents-64-*`, `agents-70-*`, `agents-71-*`, `agents-72-*`, `agents-73-*`, `agents-guard.yml`, `agents-debug-*`, `agents-keepalive-*`, `agents-moderate-*`, `agents-pr-meta-v4.yml` |
| Health | 6 | `health-40-*`, `health-41-*`, `health-42-*`, `health-43-*`, `health-44-*`, `health-50-*` |
| Maintenance | 8 | `maint-45-*`, `maint-46-*`, `maint-47-*`, `maint-50-*`, `maint-51-*`, `maint-52-*`, `maint-60-*`, `maint-coverage-guard.yml` |
| Gate/Autofix | 3 | `pr-00-gate.yml`, `autofix.yml`, `selftest-reusable-ci.yml` |

## New Pattern

The repository now uses **thin caller workflows** that reference centralized
reusable workflows in `stranske/Workflows/.github/workflows/`:

```yaml
# Example: New consumer workflow pattern
jobs:
  ci:
    uses: stranske/Workflows/.github/workflows/reusable-10-ci-python.yml@main
    with:
      python-versions: '["3.11", "3.12"]'
```

## Recovery

If needed, these files can be restored from this archive or git history.
The new consumer pattern provides:

- Centralized maintenance of workflow logic
- Automatic updates when Workflows repo is updated
- Consistency across all consumer repositories
- Reduced duplication and maintenance burden

## Reference

- Source of truth: `stranske/Workflows`
- Consumer pattern docs: `stranske/Workflows/docs/INTEGRATION_GUIDE.md`
- Setup checklist: `stranske/Workflows/docs/templates/SETUP_CHECKLIST.md`
