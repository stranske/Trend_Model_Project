## Summary
- [ ] Provide a concise description of the change.
- [ ] Note any follow-up tasks or docs to update later.

## Testing
- [ ] Listed the commands or scripts used to validate the change.
- [ ] Attached or linked relevant logs when tests are not applicable.

## CI readiness
- [ ] Skimmed the [workflow spotlight](../docs/ci/WORKFLOW_SYSTEM.md#spotlight-the-six-guardrails-everyone-touches) for Gate, Maint 46, Repo Health, Actionlint, Agents Orchestrator, and Agents Guard when touching automation.
- [ ] Reviewed the [Workflow System Overview](../docs/ci/WORKFLOW_SYSTEM.md#required-vs-informational-checks-on-phase-2-dev) to confirm Gate / `gate` is the required status and Maint 46 Post CI is informational after merge.
- [ ] Checked this pull request's **Checks** tab to confirm Gate / `gate` appears under **Required checks** (Health 45 Agents Guard auto-adds when `agents-*.yml` files change).
- [ ] Escalated via the [branch protection playbook](../docs/ci/WORKFLOW_SYSTEM.md#branch-protection-playbook) if Gate / `gate` is missing or Maint 46 Post CI shows up as required.
- [ ] Confirmed the latest Gate run is green (or linked the failing run with context).
