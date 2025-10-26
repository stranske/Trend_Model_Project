# Issue Sync agent label policy

The Agents 63 Issue Sync workflow normalizes topic lists into GitHub issues. To prevent
automation from launching against the wrong agent, the workflow now enforces manual agent
selection across the sync/bridge hand-off.

## Sync: labels from imports

* Any `agent:*` or `agents:*` labels found in the imported topic payload are stripped during
  processing. They appear in the run summary under **Agent labels stripped** so humans can
  quickly reapply the correct label once triage is complete.
* The workflow still applies non-agent labels supplied in the payload and leaves any existing
  agent assignments untouched when refreshing an issue. Manual selection happens after the
  issue lands in the repository.

## Bridge: validation before copy/paste output

* The reusable bridge workflow refuses to continue unless the target issue has **exactly one**
  `agent:*`/`agents:*` label after manual triage. Missing or multiple labels fail fast with a
  descriptive error so the run can be retriggered once the issue is corrected.
* The bridge derives the `@{agent}` mention for instructions directly from the surviving label.
  The suffix portion of the label is normalized to a lowercase slug and reused for the
  instructions, `BRIDGE_AGENT_*` environment variables, and copy/paste scaffold.

These guardrails ensure human confirmation picks the right automation before a branch or PR is
prepared, while keeping downstream prompts and instructions aligned with that selection.
