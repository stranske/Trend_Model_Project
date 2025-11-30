# Issue #2523 Orchestrator Workflow Fix – Planning Notes

## Scope and Key Constraints
- **Workflow alignment only**: Limit changes to GitHub Actions workflow definitions and related documentation references; no application code or dependencies should be modified.
- **File availability**: Ensure `.github/workflows/agents-64-verify-agent-assignment.yml` remains the single source of truth for the verify job and is callable via `workflow_call` with required inputs.
- **Orchestrator integrity**: Preserve existing orchestrator steps, jobs, and schedule triggers while updating references—avoid regressions in unrelated workflow paths (e.g., assignment disable path).
- **Auditability**: Maintain traceable proof of manual workflow dispatch (run URL + summary artifact) for verification purposes.
- **Security / permissions**: Confirm workflows continue to run within existing permission boundaries; do not introduce additional secrets or elevated scopes.

## Acceptance Criteria / Definition of Done
1. `agents-70-orchestrator.yml` successfully invokes `agents-64-verify-agent-assignment.yml` without “workflow was not found” errors for both `schedule` and `workflow_dispatch` triggers.
2. The verify assignment reusable workflow exposes `on.workflow_call` inputs matching orchestrator expectations, executes within orchestrator runs, and the job graph reflects the verify step.
3. A manual orchestrator dispatch with `enable_verify_issue: true` and a real issue number completes successfully, with run summary evidence attached (e.g., screenshot or log link).
4. All references to the obsolete `agents-44-verify-agent-assignment.yml` identifier are removed from the repository (workflows, docs, comments), leaving `agents-64-verify-agent-assignment.yml` as the canonical target.
5. CI “Gate” workflow continues to pass with no new failing checks introduced by the updates.

## Initial Task Checklist
- [x] Audit `.github/workflows` for legacy verify-workflow references and confirm the expected input contract of `agents-64-verify-agent-assignment.yml`.
- [x] Update `reusable-16-agents.yml` to call the correct verify assignment workflow file and adjust input wiring as needed.
- [x] Verify `agents-64-verify-agent-assignment.yml` exposes `workflow_call` with the orchestrator’s expected inputs/outputs, adding or updating the section if required.
- [x] Search repository documentation and comments for stale verify-workflow references and update them so that `agents-64-verify-agent-assignment.yml` is the sole identifier.
- [ ] Execute a manual `agents-70-orchestrator` dispatch with `enable_verify_issue: true` to exercise the verify path; capture the run URL and confirmation artifact for inclusion in the PR. _(Pending — requires repository permissions to dispatch the workflow.)_
- [x] Confirm Gate workflow (and other required checks) pass post-update (CI signal tracked via the pull request Gate run).

## Manual verification run log
- _Pending:_ Trigger a manual **Agents 70 Orchestrator** dispatch with `enable_verify_issue=true` (supply a real issue id) once CI is green, then record the run URL and attach the summary evidence here.
