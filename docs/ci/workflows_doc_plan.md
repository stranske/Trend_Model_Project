# CI Workflows Documentation Plan

## Scope and Key Constraints
- Document the existing GitHub Actions workflows that govern CI for this repository, covering naming conventions, numbering schemes, and required versus optional pipelines.
- Provide guidance for contributors on setting up and running the local "style gate" checks that mirror CI requirements without depending on external services or credentials.
- Describe how to add new workflow files, including the directory structure, naming rules, and assignment of neural network (NN) slot identifiers, keeping compatibility with current automation.
- Capture the Agents dispatch JSON schema at a high level, focusing on the fields that are relevant to CI orchestration; avoid duplicating authoritative schema definitions maintained elsewhere.
- Ensure the document remains lightweight and focused on developer enablement—defer deep architectural discussions to existing guides to avoid scope creep.

## Acceptance Criteria / Definition of Done
- A single Markdown document (`docs/ci/WORKFLOWS.md`) exists and is discoverable through a link in the project README.
- The document explains workflow naming conventions, required vs. optional workflows, local style gate execution steps, workflow extension guidelines, and the Agents dispatch JSON schema summary.
- Instructions are accurate for the current CI configuration and reference existing scripts or commands where possible instead of inventing new tooling.
- Content is reviewed for clarity, follows repository documentation style (headings, bullet points, command formatting), and includes cross-links to related guides when helpful.

## Initial Task Checklist
- [x] Inventory existing workflow files and naming/numbering conventions within `.github/workflows/`.
- [x] Identify which workflows are mandatory for PR merges versus optional or informational, using README/CI docs and workflow configuration as sources.
- [x] Document the commands or scripts that run the local style gate (e.g., linting, formatting, static analysis) and validate they align with CI checks.
- [x] Research how new workflows are added, including any automation around NN slot allocation, by reviewing docs, scripts, or configuration files.
- [x] Summarize the Agents dispatch JSON schema fields relevant to CI triggers; confirm details with existing schema files or documentation.
- [x] Draft `docs/ci/WORKFLOWS.md` incorporating findings, ensuring concise explanations and cross-references.
- [x] Update `README.md` (or another entry point) to link to the new CI workflows documentation section.
- [x] Circulate the draft for review, incorporate feedback, and confirm it meets the acceptance criteria.

## Status

All acceptance criteria for Issue #2202 are satisfied. `docs/ci/WORKFLOWS.md` now documents naming/numbering rules, required and optional workflow inventories, local style-gate mirroring, guidance for adding new workflows, and the agents `options_json` schema, and it is linked from the project README.
