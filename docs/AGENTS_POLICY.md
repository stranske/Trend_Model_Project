# Agents Workflow Protection Policy

Trend Model Project is a consumer of the shared automation maintained in
`stranske/Workflows`. The local contract-critical surfaces are:

- `.github/workflows/agents-issue-intake.yml`
- `.github/workflows/agents-auto-pilot.yml`
- `.github/workflows/agents-71-codex-belt-dispatcher.yml`
- `.github/workflows/agents-72-codex-belt-worker-dispatch.yml`
- `.github/workflows/agents-72-codex-belt-worker.yml`
- `.github/workflows/agents-73-codex-belt-conveyor.yml`
- `.github/workflows/agents-80-pr-event-hub.yml`
- `.github/workflows/agents-81-gate-followups.yml`
- `.github/workflows/agents-guard.yml`

The consumer-side Agents 70 orchestrator was retired. Do not recreate it or
route operators to it.

## Change Contract

1. Fix shared behavior and consumer templates in `stranske/Workflows` first.
2. Apply the matching consumer sync or an explicitly source-matched forward
   fix here; do not introduce a consumer-only fork.
3. Use the `agents:allow-change` label and obtain the reviews required by
   CODEOWNERS and branch protection for protected workflow edits.
4. Keep automation-created pull requests ready for review. Draft state and PR
   closure are not dependency, staging, stack-order, or capacity controls.
5. Verify `isDraft=false` before handing off a created or reused automation PR.

Health 45 Agents Guard enforces the protected `agents-*.yml` surface. Emergency
bypasses require a maintainer, a dedicated PR, and restoration of the guard
immediately after the change.

See [Repo Ops Facts - Agent Bootstrap](ops/codex-bootstrap-facts.md) for the
operator entry point and shared-source boundary.
