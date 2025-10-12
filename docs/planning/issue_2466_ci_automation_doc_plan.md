# CI / Automation Documentation Refresh Plan (Issue #2466)

## Scope and Key Constraints
- Refresh the CI and automation documentation set to match the simplified workflow topology that is currently in production.
- Confine updates to documentation artifacts; no workflow YAML or implementation changes are expected.
- Treat the Orchestrator workflow as the canonical automation entry point and eliminate references to deprecated flows (agent-watchdog, unconstrained agents-consumer, legacy self-test variants).
- Preserve alignment with the job names, triggers, and inputs that appear in the Actions UI today so new contributors can map docs to what they see live.
- Provide concise visuals (diagram or table) without introducing heavy diagram tooling requirements; prefer Markdown-native tables or lightweight Mermaid.
- Keep guidance consistent with existing README pointers and ensure cross-links (e.g., CONTRIBUTING.md, docs/ci/WORKFLOWS.md) are accurate after edits.

## Acceptance Criteria / Definition of Done
- `docs/ci/WORKFLOWS.md` documents only currently active workflows, enumerating for each: trigger(s), required vs optional status, permissions, and primary responsibilities.
- `docs/WORKFLOW_GUIDE.md` and `docs/ci_reuse.md` describe the Orchestrator as the single automation entry point and contain no references to agent-watchdog or unconstrained consumer flows.
- `CONTRIBUTING.md` explicitly identifies the gate workflow as the mandatory merge check and links to recommended local validation scripts (style gate, full gate).
- Documentation includes a clear representation (table or diagram) of the gate pipeline that highlights the failure reporting path (step summaries, link surfaces).
- Instructions exist for manually dispatching the Orchestrator (workflow_dispatch) that list required inputs and common optional toggles.
- A troubleshooting section covers agent bootstrap readiness signals and what contributors should verify before escalating issues.
- All hyperlinks are validated (relative or absolute) and point to existing resources in the repository.

## Initial Task Checklist
1. **Inventory Current Docs** – Review `docs/ci/WORKFLOWS.md`, `docs/WORKFLOW_GUIDE.md`, `docs/ci_reuse.md`, `CONTRIBUTING.md`, and `docs/agent-automation.md` (or related files) to catalog outdated references.
2. **Confirm Live Workflow State** – Cross-check `.github/workflows/` directory and recent Actions runs to verify active workflow names, triggers, and required checks.
3. **Define Gate Summary Artifact** – Decide on table vs Mermaid diagram for the gate pipeline, gather step names, and outline failure reporting touchpoints.
4. **Draft Orchestrator Dispatch Instructions** – Capture required inputs, optional parameters, and links/screenshots for manual workflow dispatch.
5. **Author Troubleshooting Slice** – Compile bootstrap readiness signals (logs, status checks, environment setup) and recommended remediation steps.
6. **Update Target Docs** – Apply edits to the identified files, ensuring deprecated references are removed and new guidance is inserted.
7. **Link and Consistency Audit** – Run link checks (manual or tooling) and ensure CONTRIBUTING/README cross-links remain coherent.
8. **Request Review** – Circulate the updated documentation for review with CI/docs stakeholders before merge.

