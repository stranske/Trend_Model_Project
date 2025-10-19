# Codex Task #2816 — Autofix Workflow Guardrails

Reference: [Issue #2816](https://github.com/stranske/Trend_Model_Project/issues/2816)

## Scope / Key Constraints
- Restrict automated changes to cosmetic-safe operations (Ruff `--fix`, import sorting, whitespace normalisation) within the reusable autofix workflow.
- Limit file touch set to code and test directories unless explicit whitelist exceptions exist.
- Ensure automation labels the PR outcome accurately (`autofix:applied` when commits exist, otherwise `autofix:clean`).
- Generate a concise PR comment enumerating changed files via `git diff --name-only`.

## Acceptance Criteria / Definition of Done
- Demo PR demonstrates a single commit comprised solely of safe cosmetic updates produced by the workflow.
- Workflow successfully applies the correct outcome label based on whether changes were made.
- No non-code/test files are modified unless whitelisted.
- PR comment is posted summarising touched files when changes occur.

## Initial Task Checklist
1. Update `.github/workflows/reusable-18-autofix.yml` to sequence Ruff formatting (imports, whitespace, lint fixes) as discrete steps.
2. Add logic to apply `autofix:applied` vs `autofix:clean` labels depending on change detection.
3. Incorporate a step that runs `git diff --name-only` and posts the resulting file list as a PR comment when changes exist.
4. Validate workflow on a sample branch to confirm only cosmetic adjustments are committed and labels/comments match expectations.
