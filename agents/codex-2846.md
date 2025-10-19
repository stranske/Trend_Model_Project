<!--
Context: Bootstrap file for Codex agent initialization.
Issue: https://github.com/stranske/Trend_Model_Project/issues/2846
Purpose: Track quoting fixes for the self-test reusable workflow.
Date: 2024-06-10
Owner: @codex-maintainer
-->

# Issue #2846 – Self-test reusable workflow quoting fixes

## Task Checklist
- [x] Harden selftest-reusable-ci.yml defaults and validate execution logic
  - Confirmed `${{ jobs.scenarios.uses }}` continues to point at the reusable workflow and retains the `fail-fast: false` matrix strategy.
  - Ensured the publish job still uploads the `selftest-report` artifact and appends scenario details to `$GITHUB_STEP_SUMMARY`.
- [ ] Verify workflow_dispatch run uploads expected artifacts *(pending manual GitHub Actions run)*
- [ ] Confirm nightly cron success *(awaiting scheduled execution post-merge)*

## Acceptance Criteria Status
- [ ] Manual workflow_dispatch run succeeded with artifacts *(requires GitHub Actions run)*
- [ ] Nightly cron run succeeded *(will be monitored post-merge)*
- [x] Per-scenario results summarized via GITHUB_STEP_SUMMARY

## Validation Plan
1. Trigger `Selftest: Reusables` via **Run workflow** and provide any optional inputs that should override the defaults.
2. Confirm the run uploads the `selftest-report` artifact, renders the scenario table in the run summary, and (when configured) posts the PR comment using the supplied title overrides.
3. After merging, monitor the next scheduled cron execution to ensure it completes successfully with the new quoting defaults.

## Notes
- Local environment cannot dispatch GitHub workflows; manual validation will be requested from maintainers once changes land.
- Follow up with SRE to verify the first cron completion after deployment and update the checklist accordingly.
