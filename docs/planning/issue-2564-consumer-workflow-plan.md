# Issue #2564 — Consumer workflow retirement (archived)

> **Status:** Completed 2026-02-08. The legacy consumer wrappers (Agents 61/62)
> were permanently removed in favour of the orchestrator-only topology. This
> note remains solely to record the outcome; do not recreate the retired
> workflows.

## Outcome summary
- `agents-61-consumer-compat.yml` and `agents-62-consumer.yml` were deleted; all
  manual dispatches must use `agents-70-orchestrator.yml` instead.
- Orchestrator documentation and guardrails (policy, workflow system overview,
  tests) were updated to describe the Agents 63 + 70 topology as the single
  supported surface.
- Follow-up hygiene tasks monitor the repository for any reintroduction of the
  retired slugs and redirect consumers to the orchestrator entry point.

## Historical log
- **2025-10-14** — The workflows briefly remained as manual shims while the
  orchestrator rollout stabilized. That decision is now obsolete.
- **2026-02-08** — Issue #2682 finalized the retirement by removing the files
  and updating documentation and guardrails accordingly.

