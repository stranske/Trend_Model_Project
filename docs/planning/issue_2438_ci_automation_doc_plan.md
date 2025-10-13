# Issue #2438 – CI & Automation Documentation Update Plan

## Scope and Key Constraints
- Update the `CONTRIBUTING.md` guide with a new "CI & Automation" section that documents the streamlined workflow topology for pull requests.
- Limit edits to documentation only; no workflow YAMLs or automation scripts should be modified.
- Ensure all workflow names, file paths, and labels reflect the current repository configuration (Gate required check, consolidated Maint Post-CI, the Agents 70 Orchestrator workflow, and reusable includes).
- Preserve existing CONTRIBUTING.md structure and tone; integrate the new section without disrupting other guidance.
- Provide accurate references to automation behavior such as gate enforcement, autofix timing/opt-outs, and agent-triggering labels.

## Acceptance Criteria / Definition of Done
- `CONTRIBUTING.md` contains a clearly titled "CI & Automation" section.
- Section explicitly calls out:
  - The required Gate check ("Gate" / `gate`).
  - When the autofix workflow executes, what it does, and how contributors opt in or out.
  - How agent automations (bootstrap, readiness, watchdog) are triggered via the orchestrator, including the relevant labels and their effects.
  - Links to the workflow YAML files: `agents/agents-70-orchestrator.yml` and `reusable-16-agents.yml`.
  - Note that PR coverage/status are summarized in a single consolidated Maint Post-CI comment.
- Language is consistent with repository documentation standards and free of stale references.
- Markdown renders correctly (proper headings, lists, and links).
- PR passes Gate (documentation-only change should still satisfy the required check).

## Initial Task Checklist
1. [x] Review existing documentation to confirm where a "CI & Automation" section best fits within `CONTRIBUTING.md`.
2. [x] Audit the latest workflow files to verify names, labels, and behaviors (Gate, Maint Post-CI, orchestrator, reusable includes, autofix opt-in/out).
3. [x] Draft the new section content covering required points (Gate, autofix timing & opt-in/out, agent automation triggers, consolidated Maint Post-CI summary).
4. [x] Add inline links to the relevant workflow YAML files and ensure paths are correct.
5. [x] Proofread for clarity, consistency, and adherence to documentation tone/style.
6. [x] Run formatting or linting checks if applicable for Markdown (none expected, but ensure no trailing whitespace and valid Markdown syntax).
7. [x] Commit changes, run the Gate check if needed, and submit PR for review.
