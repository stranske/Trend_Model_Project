# CI Documentation Alignment Plan (Issue #2497)

## Scope and Key Constraints
- Update CI/automation documentation (including `docs/ci/WORKFLOWS.md`, `docs/WORKFLOW_GUIDE.md`, `docs/ci_reuse.md`, and `README.md`) to reflect the final workflow topology, filenames, and triggers described in issue #2497.
- Limit changes to documentation artifacts; no workflow YAML or automation logic updates are in scope.
- Treat the Orchestrator reusable as the single entry point for automation and clearly mark any remaining legacy flows (agent-watchdog, legacy consumer) as deprecated references only.
- Ensure that all references to reusable workflow filenames and `pr-00-gate.yml` match their live, renamed counterparts.
- Provide lightweight visuals (Markdown tables or Mermaid diagram) without introducing heavyweight tooling dependencies.
- Document manual dispatch of the Orchestrator via `workflow_dispatch`, including JSON payload examples for `params_json`.
- Include troubleshooting guidance for the repo-health check, emphasizing permission prerequisites and escalation steps.

## Acceptance Criteria / Definition of Done
- `docs/ci/WORKFLOWS.md` lists only active workflows with accurate filenames, triggers, required/optional status, and permissions.
- `docs/WORKFLOW_GUIDE.md` and `docs/ci_reuse.md` present the Orchestrator as the sole automation entry point, with legacy components referenced only as deprecated (if at all).
- `README.md` correctly references the renamed reusable workflows and `pr-00-gate.yml`.
- A new or updated table/diagram describes the Gate workflow, covering jobs, produced artifacts, and the final enforcement job.
- Documentation includes clear instructions for manually triggering the Orchestrator, including a `params_json` example payload.
- Troubleshooting notes address repo-health permission failures with actionable remediation steps.
- Cross-links among updated documents remain valid and point to existing resources.

## Initial Task Checklist
1. **Assess Current Documentation** – Review existing CI docs (`docs/ci/WORKFLOWS.md`, `docs/WORKFLOW_GUIDE.md`, `docs/ci_reuse.md`, `README.md`) to catalogue outdated references and identify sections requiring updates.
2. **Validate Live Workflow Inventory** – Confirm active workflow filenames, triggers, and required checks by inspecting `.github/workflows/` and recent GitHub Actions runs.
3. **Design Gate Visualization** – Decide on table vs. diagram for gate workflow representation and gather job/artifact details to include.
4. **Draft Orchestrator Dispatch Guidance** – Document manual trigger steps, required inputs, and craft a sample `params_json` payload illustrating common parameters.
5. **Author Repo-Health Troubleshooting** – Outline permission requirements, typical failure symptoms, and remediation/escalation guidance for the repo-health check.
6. **Update Target Documents** – Apply edits across the identified documentation files, ensuring terminology and references align with the final workflow topology.
7. **Perform Consistency Review** – Verify internal/external links, filenames, and cross-document references; ensure README and CI docs stay synchronized.
8. **Stakeholder Review** – Share the updated documentation with CI/doc stakeholders for validation before merge.
