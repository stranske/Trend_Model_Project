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
  - Re-installed `actionlint` (v1.7.8) locally and re-ran `/root/.local/share/mise/installs/go/1.23.8/bin/actionlint .github/workflows/selftest-reusable-ci.yml` to confirm the workflow parses successfully (2025-02-14).
- [ ] Verify workflow_dispatch run uploads expected artifacts *(pending manual GitHub Actions run)*
  - [ ] Coordinate maintainer-triggered dispatch once branch is ready for validation.
  - [x] Drafted maintainer request snippet for the manual dispatch (see **Maintainer Outreach**).
- [ ] Confirm nightly cron success *(awaiting scheduled execution post-merge)*
  - [ ] Schedule follow-up reminder after manual dispatch completes to capture the first successful cron evidence.

## Acceptance Criteria Status
- [ ] Manual workflow_dispatch run succeeded with artifacts *(awaiting maintainer-run; see Run Tracking)*
- [ ] Nightly cron run succeeded *(monitor after merge; see Run Tracking)*
- [x] Per-scenario results summarized via GITHUB_STEP_SUMMARY

## Validation Plan
0. Pre-flight: run `actionlint` locally on `.github/workflows/selftest-reusable-ci.yml` to ensure expression syntax remains valid (**completed**, see checklist).
1. Trigger `Selftest: Reusables` via **Run workflow** and provide any optional inputs that should override the defaults.
2. Confirm the run uploads the `selftest-report` artifact, renders the scenario table in the run summary, and (when configured) posts the PR comment using the supplied title overrides.
3. After merging, monitor the next scheduled cron execution to ensure it completes successfully with the new quoting defaults.

## Notes
- Local environment cannot dispatch GitHub workflows; manual validation will be requested from maintainers once changes land.
- Follow up with SRE to verify the first cron completion after deployment and update the checklist accordingly.
- Added local tooling notes below to document the environment used for syntax validation.

## Local Validation Evidence
- 2025-02-14: Installed `actionlint` (v1.7.8) via `go install` and executed `/root/.local/share/mise/installs/go/1.23.8/bin/actionlint .github/workflows/selftest-reusable-ci.yml` (no findings).

## Run Tracking
| Path | Status | Run link / ID | Notes |
| ---- | ------ | ------------- | ----- |
| `workflow_dispatch` | Pending (blocked on maintainer run) | _TBD_ | Request drafted; awaiting maintainer bandwidth to execute validation checklist and record artifact links. |
| Nightly cron | Pending | _Next available schedule_ | Will monitor after merge once manual verification establishes baseline. |

## Next Steps
- Draft maintainer request for manual dispatch including validation checklist and artifact expectations.
- Capture run metadata (run ID, artifact links, summary screenshot) once execution completes and update the acceptance criteria and table above.
- Schedule follow-up reminder to confirm the first cron run after merge and record outcome in **Run Tracking**.

## Maintainer Outreach
```
@dev-infra Could someone trigger **Selftest: Reusables** (`selftest-reusable-ci.yml`) against this branch via workflow_dispatch?

Please capture:
- Artifact: `selftest-report`
- Run summary snippet showing the scenario table and the `${{ inputs.summary_title }}` override if one is supplied
- Overall run conclusion

Once complete, drop the run URL so we can update the tracker and mark the acceptance criteria.
```
