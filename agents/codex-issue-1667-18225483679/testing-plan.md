# Testing Plan

## Automated Coverage
- Run `pytest tests/test_workflow_merge_manager.py` to exercise the merge-manager guardrails.
- If workflow fixtures change, execute `pytest tests/test_trend_cli_entrypoints.py` to ensure CLI entry points still load without the removed fonts.
- Lint the repository with `black` and `ruff` (as invoked by CI) before opening a PR to avoid formatter failures.

## Manual Validation
1. Create a documentation-only PR with the `automerge` label. Verify the Merge Manager comment shows all requirements satisfied once CI and Docker pass.
2. Remove the `automerge` label and confirm the Merge Manager leaves the PR pending with a `missing automerge label` reason.
3. Apply a label containing `breaking` (e.g., `breaking-change`) and ensure automerge stays disabled with the expected rationale.
4. Inspect the workflow logs to verify CI and Docker run identifiers are linked to the head commit.

## Regression Checks
- Confirm `src/trend/reporting/unified.py` continues to generate PDFs successfully without bundled binary fonts by running `pytest tests/test_unified_report.py`.
- Re-run the Merge Manager workflow in dry-run mode (via `act` or the workflow simulator) whenever status logic changes to guarantee parity between docs and automation.
