# Gate Workflow Branch Protection Plan

## Scope and Key Constraints
- Enforce the `gate` GitHub Actions workflow as a required status check on the default branch (currently `main`).
- Remove any legacy required status checks that overlap or conflict with `gate` (e.g., older `CI` workflows).
- Enable the "Require branches to be up to date" option so merges must include the latest default-branch commits.
- Preserve existing job names inside `gate.yml` (`core tests (3.11)`, `core tests (3.12)`, `docker smoke`, `gate`) to avoid downstream automation regressions.
- Communicate the new requirement in contributor-facing docs (CONTRIBUTING.md) without altering unrelated guidance.

## Acceptance Criteria / Definition of Done
- Default branch protection rule lists `gate` as a required status check and prevents merging until it succeeds.
- Any deprecated required checks (e.g., `CI`) are removed from the protection rule.
- "Require branches to be up to date" is enabled for the default branch protection rule.
- CONTRIBUTING.md explicitly instructs contributors that the `gate` check must pass before merging.
- A test pull request shows the `gate` check as required and blocks merge when failing or pending.

## Initial Task Checklist
1. Audit current branch protection settings for the default branch and note existing required checks.
2. Update branch protection:
   - Add `gate` as a required status check.
   - Remove obsolete required checks.
   - Enable "Require branches to be up to date" if not already set.
3. Verify recent `gate` workflow runs to confirm job naming and stability.
4. Create or update documentation (CONTRIBUTING.md) to mention the required `gate` check.
5. Open a validation pull request to confirm the `gate` check appears as required and blocks merge until passing.
6. Record findings and screenshots/logs (if applicable) demonstrating the protection rule and validation PR behavior.
