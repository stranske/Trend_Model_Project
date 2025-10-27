<!-- bootstrap for Codex on issue #2684: https://github.com/stranske/Trend_Model_Project/issues/2684 -->

# Execution Plan

## Scope / Key Constraints
- Produce a new `docs/ci/AGENTS_POLICY.md` page that centralizes the contract for the Agents 63 pair (issue bridge + ChatGPT sync) and the Agents 70 orchestrator.
- Document the protections that render these workflows "unremovable," including CODEOWNERS, the repository ruleset, and the Agents Guard workflow, without altering any enforcement configuration.
- Capture the narrow, allow-listed conditions that justify modifying the protected workflow files and describe how the `agents:allow-change` label and maintainer review fit into the process.
- Provide quick-start troubleshooting guidance focused on common policy enforcement failures (e.g., missing label, CODEOWNERS review, Agents Guard failures) while keeping the doc concise.
- Update existing references only where required: the workflow system overview document and the headers of the two Agents 63 workflow files should link to the new policy. Avoid broader refactors or unrelated documentation edits.

## Acceptance Criteria / Definition of Done
- `docs/ci/AGENTS_POLICY.md` exists with clear sections covering scope & purpose, protection layers, allowed change scenarios with the labeling workflow, and at least one actionable troubleshooting block.
- `docs/ci/WORKFLOW_SYSTEM.md` references the new policy file when discussing agents workflow protections.
- `.github/workflows/agents-63-chatgpt-issue-sync.yml` and `.github/workflows/agents-63-issue-intake.yml` include header comments linking directly to the policy doc.
- All new and updated documentation follows the existing tone/formatting conventions for CI docs (concise headings, sentence-case bullets, Markdown lint friendly).
- No functional workflow behavior changes are introduced; CI should continue to treat this as a docs-only update.

## Initial Task Checklist
- [x] Draft `docs/ci/AGENTS_POLICY.md` with sections for purpose/scope, protections, allowed change reasons & label process, and troubleshooting tips.
- [x] Cross-link the new policy from `docs/ci/WORKFLOW_SYSTEM.md` in the policy/enforcement area.
- [x] Add a top-of-file comment in each Agents 63 workflow file pointing to the policy URL.
- [x] Self-review for consistency with CI documentation voice and verify Markdown formatting/lint expectations.
