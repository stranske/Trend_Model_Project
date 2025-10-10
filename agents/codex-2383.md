# Issue #2383 — CI/Agents Workflow Catalog Planning

## Scope and Key Constraints
- Produce or update `docs/ci/WORKFLOWS.md` with a concise catalog that covers every workflow listed in the source issue (Gate, CI, Docker smoke, Autofix, Repo Health, Agents Consumer, Reuse-agents).
- Summarise per-workflow details strictly to: purpose, trigger(s), required secrets/permissions, and resulting status checks or labels. Keep sections brief for contributor readability.
- Cross-reference existing documentation (`ARCHIVE_WORKFLOWS.md`, prior CI docs) to ensure accuracy without duplicating exhaustive change logs.
- Link the catalog from `README.md` in a way that fits the existing documentation structure and linting rules (respect markdown style and link conventions already used in README).
- Maintain repository documentation style conventions (headings, ordered lists) and avoid modifying unrelated files or workflow definitions.

## Acceptance Criteria / Definition of Done
1. `docs/ci/WORKFLOWS.md` exists (or is updated) and contains individual subsections for each required workflow with:
   - A clear description of what the workflow verifies or accomplishes.
   - Specific trigger conditions (manual, push, PR, schedule, reusable call, etc.).
   - Secrets/permissions explicitly enumerated (note if none required).
   - The status check, label, or other signals the workflow applies back to PRs/issues.
2. README includes a prominent link to the workflow catalog in the documentation section without breaking existing formatting or tests (e.g., Markdown lint, doc build if applicable).
3. Any references to external docs (ARCHIVE_WORKFLOWS.md, CI docs) remain valid, with optional short pointers rather than redundant content.
4. All documentation builds or lint checks (if configured) pass locally or in CI.
5. Changes reviewed/approved with no outstanding TODOs or placeholders left in the new doc.

## Initial Task Checklist
- [ ] Audit existing documentation (README, ARCHIVE_WORKFLOWS.md, other CI docs) to collect authoritative workflow details.
- [ ] Draft the structure of `docs/ci/WORKFLOWS.md` ensuring one section per workflow and consistent formatting.
- [ ] Populate each section with the four required data points (purpose, triggers, secrets/permissions, statuses/labels) using current workflow configurations as the source of truth.
- [ ] Add cross-links to related docs where helpful (e.g., archive or deeper CI guides) while keeping the catalog concise.
- [ ] Update README with a link to the new catalog entry point, following existing Markdown style.
- [ ] Run applicable documentation or linting checks to confirm no regressions.
- [ ] Request review and incorporate feedback; confirm acceptance criteria are met before merge.
