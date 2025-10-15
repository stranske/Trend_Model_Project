# Health 45 Agents Guard – Planning Notes

## Scope and Key Constraints
- Implement a `.github/workflows/health-45-agents-guard.yml` workflow that runs on `pull_request_target` events limited to changes inside `.github/workflows/agents-*.yml`.
- Workflow must execute from the default branch context and use only `contents: read` and `pull-requests: write` permissions; no additional scopes or secrets.
- Guarded files: both "Agents 63" workflow files and the "Orchestrator" workflow/file (exact filenames to be confirmed from repository history). Deletions, renames, or missing files must cause a failure.
- Modifications to guarded files are allowed only when the PR has the `agents:allow-change` label **and** at least one CODEOWNER approval.
- The workflow should post a single, human-friendly failure comment explaining how to proceed, avoiding duplicate comments across runs.
- The failure should block merging by marking the status check as failed, and the check must be added to the repository’s required status checks list.
- Solution must rely on GitHub API interactions available in GitHub Actions and should remain compatible with forks (no write access to repo contents beyond comments).

## Acceptance Criteria / Definition of Done
1. Workflow triggers only for PRs that touch `.github/workflows/agents-*.yml` files.
2. Workflow fails immediately with an explanatory comment if any guarded file is deleted or renamed in a PR.
3. Workflow fails with an explanatory comment when guarded files are modified without the `agents:allow-change` label.
4. Workflow fails with an explanatory comment when guarded files are modified without at least one CODEOWNER approval, even if the label is present.
5. Workflow passes when guarded files are modified, the `agents:allow-change` label is present, and at least one CODEOWNER approval exists.
6. Failure comment appears only once per PR and includes guidance on resolving each guard condition.
7. Repository owners can add the workflow’s status check to required checks without additional configuration.

## Initial Task Checklist
- [ ] Inventory the exact filenames for the "Agents 63" workflows and the orchestrator to ensure the guard targets the correct paths.
- [ ] Design the GitHub Actions workflow structure (trigger, permissions, job layout) and choose the tooling (e.g., `actions/github-script` vs. custom action).
- [ ] Implement logic to fetch changed files via the GitHub API and detect deletions/renames affecting guarded files.
- [ ] Implement guard logic that evaluates label presence and CODEOWNER approvals.
- [ ] Add idempotent PR commenting to explain failures without duplication.
- [ ] Test the workflow behavior using workflow dry-runs or mock PR scenarios (e.g., `act` or manual triggering) to validate each acceptance criterion.
- [ ] Coordinate with repository settings to add the new status check to required checks after verification.
