# Maint Post-CI Consolidation Plan (Issue #2433)

## Scope / Key Constraints
- Replace the existing Maint 30 Post CI Summary and Maint 32 Autofix workflows with a single `.github/workflows/maint-30-post-ci.yml` triggered by `workflow_run` on the Gate workflow when targeting pull requests.
- Preserve current behavior for coverage artifact collection, autofix execution, JSON diagnostics, and fork safeguards; any logic moved must remain functionally equivalent.
- Emit exactly one persistent PR comment per run that consolidates Gate job status, coverage deltas/snippets, and autofix results, using stable comment identifiers to support updates.
- Upload coverage and autofix artifacts once per run using consistent names (e.g., `coverage-summary.md`, `autofix_report.json`) so downstream consumers remain compatible.
- Maintain compatibility with permission constraints (e.g., `pull_request` vs. `pull_request_target`, fork contributions) and ensure autofix respects existing guardrails (safe paths, patch caps, review gating).
- Disable or remove the legacy Maint 30 and Maint 32 workflows only after the new workflow demonstrates parity in test runs and required stakeholders sign off.

## Acceptance Criteria / Definition of Done
- A new workflow file `.github/workflows/maint-30-post-ci.yml` exists, triggered via `workflow_run` from the Gate workflow’s `completed` event and scoped to PR runs.
- The workflow produces both a summary job (always) and an autofix job (conditional on autofix eligibility) with any optional failure-tracking behavior preserved.
- Each run writes or updates a single PR comment containing: overall required-job status, coverage summary (and trend when available), and an autofix summary when changes were attempted/applied.
- Coverage summaries and autofix diagnostics are uploaded once with predictable artifact names that align with existing tooling expectations.
- Legacy workflows `maint-30-post-ci-summary.yml` and `maint-32-autofix.yml` are removed or disabled, with documentation/tests showing the consolidated workflow covers their responsibilities.
- Gate and related required checks continue to pass in CI for representative branches after consolidation, and no duplicate or missing comments occur in PRs.

## Initial Task Checklist
- [x] Audit the current Maint 30 and Maint 32 workflows to catalog all steps, conditionals, secrets, and artifacts that must be preserved.
- [x] Draft `.github/workflows/maint-30-post-ci.yml`, wiring `workflow_run` triggers, job dependencies, and shared environment variables/secrets.
- [x] Port the post-CI summary logic into a `summarize` job, ensuring coverage extraction, artifact handling, and comment composition are idempotent.
- [x] Integrate autofix logic into an `autofix` job that reuses existing guards (fork detection, safe path lists, patch caps) and emits diagnostics under the agreed artifact name.
- [x] Implement the unified PR comment writer (single comment keyed by identifier) that merges status, coverage, and autofix sections.
- [x] Upload coverage and autofix artifacts once using the standardized filenames and confirm downstream automation recognizes them.
- [ ] Test the consolidated workflow via dry-run or targeted branch runs to verify parity (including fork scenarios and failing Gate cases).
- [x] Remove or disable the legacy Maint 30/32 workflows and update documentation or references pointing to the old files.
- [ ] Confirm final CI runs show the new workflow passing and generating the expected single PR comment before merging.

> ✅ Implementation note: Maint 32’s behaviors now live in two jobs (`small-fixes` for hygiene updates and `fix-failing-checks` for lint-only failures) whose outputs roll into the unified comment. Pending items track validation of the consolidated workflow in CI.
