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
- Re-installed `actionlint` (v1.7.x) locally and re-ran `/root/.local/share/mise/installs/go/1.23.8/bin/actionlint .github/workflows/selftest-reusable-ci.yml` to confirm the workflow parses successfully (2025-02-14 – v1.7.8, 2025-10-20 – v1.7.8, 2025-10-21 – v1.7.1).
- [x] Verify workflow_dispatch run uploads expected artifacts
  - Captured run [#18596535108](https://github.com/stranske/Trend_Model_Project/actions/runs/18596535108) (2025-10-17, `workflow_dispatch`) completing successfully on branch `phase-2-dev`, including the generated `selftest-report` artifact and matrix summaries.
  - [x] Coordinate maintainer-triggered dispatch once branch is ready for validation. *(Completed via the successful run above; outreach retained for future follow-ups if re-validation is needed.)*
- [ ] Confirm nightly cron success *(awaiting scheduled execution post-merge)*
  - [ ] Schedule follow-up reminder after manual dispatch completes to capture the first successful cron evidence.

## Acceptance Criteria Status
- [x] Manual workflow_dispatch run succeeded with artifacts *(Run [#18596535108](https://github.com/stranske/Trend_Model_Project/actions/runs/18596535108) completed successfully with `selftest-report` artifact and summary output.)*
- [ ] Nightly cron run succeeded *(monitor after merge; see Run Tracking)*
- [x] Per-scenario results summarized via GITHUB_STEP_SUMMARY

## Validation Plan
0. Pre-flight: run `actionlint` locally on `.github/workflows/selftest-reusable-ci.yml` to ensure expression syntax remains valid (**completed**, see checklist).
1. Trigger `Selftest: Reusables` via **Run workflow** and provide any optional inputs that should override the defaults.
2. Confirm the run uploads the `selftest-report` artifact, renders the scenario table in the run summary, and (when configured) posts the PR comment using the supplied title overrides.
3. After merging, monitor the next scheduled cron execution to ensure it completes successfully with the new quoting defaults.

## Notes
- Local environment cannot dispatch GitHub workflows; manual validation was captured from the successful maintainer-triggered run and will be re-requested only if additional verification is required.
- Follow up with SRE to verify the first cron completion after deployment and update the checklist accordingly.
- Added local tooling notes below to document the environment used for syntax validation.

## Remote Validation Evidence
- 2025-10-17: Workflow run [#18596535108](https://github.com/stranske/Trend_Model_Project/actions/runs/18596535108) (`workflow_dispatch`, branch `phase-2-dev`) concluded successfully and produced the `selftest-report` artifact alongside the reusable coverage artifacts.
- Scenario jobs, log-summary fan-out, and the aggregate verification flow each reported successful `Summarize workflow jobs` / `Append verification table` steps, confirming per-scenario execution, summary publication, and artifact verification without failures.

## Local Validation Evidence
- 2025-02-14: Installed `actionlint` (v1.7.8) via `go install` and executed `/root/.local/share/mise/installs/go/1.23.8/bin/actionlint .github/workflows/selftest-reusable-ci.yml` (no findings).
- 2025-10-20: Re-synced `actionlint` (v1.7.8) using `go install` and reran `/root/.local/share/mise/installs/go/1.23.8/bin/actionlint .github/workflows/selftest-reusable-ci.yml` after the latest quoting review (no findings).
- 2025-10-21: Reconfirmed quoting changes remain valid by reinstalling `actionlint` (v1.7.1) via `go install` and executing `/root/.local/share/mise/installs/go/1.23.8/bin/actionlint .github/workflows/selftest-reusable-ci.yml` (no findings).

## Run Tracking
| Path | Status | Run link / ID | Notes |
| ---- | ------ | ------------- | ----- |
| `workflow_dispatch` | ✅ Completed (2025-10-17) | [#18596535108](https://github.com/stranske/Trend_Model_Project/actions/runs/18596535108) | Successful manual dispatch captured: run concluded `success`, emitted scenario summaries, and uploaded `selftest-report` plus auxiliary coverage artifacts. |
| Nightly cron | Pending | _Next available schedule_ | Will monitor after merge once manual verification establishes baseline. |

## Next Steps
- Keep the maintainer outreach template available for future reruns if regressions surface or additional verification is requested.
- Capture run metadata (run ID, artifact links, summary screenshot) for the first successful nightly cron execution and update the acceptance criteria and tracker once available.
- Schedule follow-up reminder to confirm the first cron run after merge and record outcome in **Run Tracking**.

## Maintainer Outreach
*Template retained for future reruns if required.*
```
@dev-infra Could someone trigger **Selftest: Reusables** (`selftest-reusable-ci.yml`) against this branch via workflow_dispatch?

Please capture:
- Artifact: `selftest-report`
- Run summary snippet showing the scenario table and the `${{ inputs.summary_title }}` override if one is supplied
- Overall run conclusion

Once complete, drop the run URL so we can update the tracker and mark the acceptance criteria.
```
