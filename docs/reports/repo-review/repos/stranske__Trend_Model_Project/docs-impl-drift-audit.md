# Documentation Implementation Drift Audit

Source issue: https://github.com/stranske/Trend_Model_Project/issues/5292

## Scan Command

```bash
rg --files-with-matches -g '*.md' '(\[ \]|Pending|TODO|POST /|GET /|Status:)' docs/
```

## Authoritative Implementation References

- REST routes: `src/trend_analysis/api_server/__init__.py` registers `GET /health`, `GET /`, `POST /config/patch`, and `POST /config/patch/preview`. FastAPI also exposes framework routes `GET /docs`, `GET /redoc`, and `GET /openapi.json`.
- Monte Carlo implementation: `src/trend_analysis/monte_carlo/` contains scenario/config loading, registry/discovery, bootstrap/regime models, strategy variants/sampling, runner, cache, costs, folds, aggregation, and export modules.
- Coverage thresholds: `.github/workflows/pr-00-gate.yml` and `.github/workflows/ci.yml` set `coverage-min` to `80`.

## Findings Summary

- Confirmed drift already covered by converged per-instance issues: `docs/phase-3/MonteCarlo.md`.
- New confirmed drift filed during this audit: `docs/issues/raise_test_coverage_to_89.md` -> https://github.com/stranske/Trend_Model_Project/issues/5296.
- No additional source-code changes are included in this audit PR.

## Matched Files

| File | Matched claim sites | Disposition | Rationale / follow-up |
| --- | --- | --- | --- |
| `docs/AGENT_ISSUE_FORMAT.md` | checklist examples | already-correct | Format guide intentionally demonstrates unchecked task/acceptance syntax; not an implementation-state claim. |
| `docs/debugging/keepalive_iteration_log.md` | pending/skipped historical log entries | already-correct | Debugging log records historical workflow observations, not current product or endpoint status. |
| `docs/phase-3/MonteCarlo.md` | line 3, lines 755-782 | confirmed-drift/already-covered | File says implementation is pending and all milestones unchecked, but `src/trend_analysis/monte_carlo/` has shipped modules for the listed milestones. Covered by the existing Monte Carlo stale-status per-instance issue. |
| `docs/fixes/API_RATE_LIMIT_FIX_STATUS.md` | status and merge checklist | needs-investigation | Operational fix status references old PR workflow state; it is outside product/API docs and should be reviewed with workflow-history context before filing a product drift issue. |
| `docs/quarantine_ttl_monitoring.md` | retired workflow status | already-correct | The file explicitly says the workflow was retired and keeps notes for historical reference. |
| `docs/contracts/agent-runner-output.md` | implementation checklist | already-correct | Canonical contract includes checklist items for future agent implementers; unchecked items are template requirements. |
| `docs/api-usage-diagnostic-2026-02-07.md` | healthy status section | already-correct | Dated diagnostic report; no current endpoint or implementation contract is asserted. |
| `docs/templates/AGENT_ISSUE_TEMPLATE.md` | checklist examples | already-correct | Template file intentionally contains placeholder unchecked tasks. |
| `docs/issues/raise_test_coverage_to_89.md` | line 3, line 16 | confirmed-drift | The document says current threshold is `74%`; current Gate/CI workflows use `80`. Filed follow-up issue https://github.com/stranske/Trend_Model_Project/issues/5296. |
| `docs/archive/plans/issue-2523-plan.md` | pending manual dispatch items | already-correct | Archived plan; historical pending state is preserved intentionally. |
| `docs/fastapi-migration.md` | lines 70-74 | already-correct | Endpoint list matches current app plus FastAPI framework docs routes; it does not claim the phantom `/analyze` endpoint. |
| `docs/evidence/agents-orchestrator/manual-run-2025-10-14.md` | pending external execution | already-correct | Evidence note is historical and scoped to a manual workflow run. |
| `docs/evidence/agents-orchestrator/manual-run-issue-2566.md` | completed/pass status | already-correct | Dated evidence file; no current product implementation claim. |
| `docs/DEPENDENCY_MANAGEMENT.md` | system status | already-correct | Current policy/status doc; matched status is not contradicted by audited implementation references. |
| `docs/archive/plans/coverage_trend_plan.md` | historical unchecked task list | already-correct | Archived plan preserves prior task state. |
| `docs/UI_Parameter_Audit.md` | complete status entries | already-correct | Completed audit report; matched statuses are report results, not implementation promises requiring code cross-check. |
| `docs/archive/plans/issues-3260-3261-keepalive-log.md` | unchecked validation rows | already-correct | Archived keepalive log; historical status tracking. |
| `docs/monte_carlo/visualization.md` | example unchecked task text | already-correct | Example text for follow-up issue creation, not a claim about current implementation. |
| `docs/archive/plans/coverage_progress.md` | coverage checklist | already-correct | Archived coverage plan; historical triage list. |
| `docs/archive/plans/selftest_manual_plan.md` | unchecked manual-run reminder | already-correct | Archived plan item, not current implementation docs. |
| `docs/archive/plans/DEPENDENCY_MANAGEMENT_SUMMARY.md` | complete status | already-correct | Archived summary states a completed historical migration. |
| `docs/archive/plans/repo-health-self-check-plan.md` | pending manual validation | already-correct | Archived plan; historical/manual workflow status. |
| `docs/settings_evidence/rank_pct.md` | FAIL status | already-correct | Generated setting-evidence file; status reflects audit result for that setting. |
| `docs/settings_evidence/lookback_periods.md` | PASS status | already-correct | Generated setting-evidence result. |
| `docs/archive/plans/agent_assignment_verification_plan.md` | REST API call example | already-correct | GitHub API endpoint example, not this repo's FastAPI server. |
| `docs/archive/plans/issue-3261-keepalive-detection-log.md` | pending validation entries | already-correct | Archived keepalive detection log. |
| `docs/ci/maint-post-ci-consolidation-plan.md` | pending CI validation items | already-correct | CI maintenance plan; not product/API implementation docs. |
| `docs/archive/plans/issue-2683-branch-protection-plan.md` | branch protection checklist | already-correct | Archived governance plan. |
| `docs/settings_evidence/z_exit_soft.md` | PASS status | already-correct | Generated setting-evidence result. |
| `docs/archive/plans/issue-2528-doc-alignment-plan.md` | unchecked planning tasks | already-correct | Archived doc-alignment plan. |
| `docs/ci/branch_protection_plan.md` | pending manual dispatch | already-correct | CI governance plan; manual validation status is explicit. |
| `docs/settings_evidence/risk_target.md` | PASS status | already-correct | Generated setting-evidence result. |
| `docs/archive/plans/selftest_81_reusable_ci_plan.md` | checked TODO note | already-correct | Archived self-test plan. |
| `docs/ci/ISSUE_FORMAT_GUIDE.md` | issue-template checklist examples | already-correct | Format guide intentionally contains sample unchecked task syntax. |
| `docs/archive/plans/issue-3260-keepalive-validation-log.md` | pending revalidation rows | already-correct | Archived validation log. |
| `docs/archive/plans/health-40-repo-selfcheck-plan.md` | workflow validation checklist | already-correct | Archived plan; historical workflow validation. |
| `docs/settings_evidence/selection_count.md` | FAIL status | already-correct | Generated setting-evidence result. |
| `docs/ci/mypy-pinning-plan.md` | planning checklist | already-correct | CI plan, not current product/API docs. |
| `docs/settings_evidence/transaction_cost_bps.md` | FAIL status | already-correct | Generated setting-evidence result. |
| `docs/ci/MERGE_QUEUE.md` | pending checks wording | already-correct | Generic process documentation; no contradiction found. |
| `docs/pr-reviews/PR4383_bot_comments_evaluation.md` | optional follow-up checklist | already-correct | PR review evaluation log. |
| `docs/settings_evidence/weighting_scheme.md` | FAIL status | already-correct | Generated setting-evidence result. |
| `docs/settings_evidence/mp_max_funds.md` | PASS status | already-correct | Generated setting-evidence result. |
| `docs/settings_evidence/max_weight.md` | FAIL status | already-correct | Generated setting-evidence result. |
| `docs/settings_evidence/min_weight.md` | PASS status | already-correct | Generated setting-evidence result. |
| `docs/settings_evidence/trend_window.md` | FAIL status | already-correct | Generated setting-evidence result. |
| `docs/TMP_TRANSITION_PLAN.md` | migration checklists | needs-investigation | Large transition plan mixes setup requirements with historical state; no single concrete product/API drift was confirmed in this audit. |
| `docs/settings_evidence/inclusion_approach.md` | PASS status | already-correct | Generated setting-evidence result. |
| `docs/workflows/SystemEvaluation.md` | scope limitation note | already-correct | Explicit limitation statement, not a false implementation claim. |
| `docs/quickstart.md` | user checklist | already-correct | End-user checklist for manual setup and run completion. |
| `docs/settings_evidence/vol_floor.md` | PASS status | already-correct | Generated setting-evidence result. |
| `docs/settings_evidence/z_entry_soft.md` | PASS status | already-correct | Generated setting-evidence result. |
| `docs/settings_evidence/leverage_cap.md` | FAIL status | already-correct | Generated setting-evidence result. |
| `docs/workflows/WorkflowSystemBugReport.md` | issue checklist | already-correct | Bug-report checklist template, not current implementation status. |
| `docs/planning/issue-2560-orchestrator-workflow-plan.md` | pending manual dispatch | already-correct | Historical planning file. |
| `docs/planning/issue-2564-consumer-workflow-plan.md` | completed status | already-correct | Historical planning file. |
| `docs/reports/repo-review/repos/stranske__Trend_Model_Project/docs-impl-drift-audit.md` | audit command and endpoint evidence | already-correct | This generated audit report contains the scan pattern and implementation evidence that intentionally repeat matched terms. |

## Confirmed Drift Details

### `docs/phase-3/MonteCarlo.md`

Disposition: `confirmed-drift/already-covered`

Evidence: the document still marks the framework as pending and lists unchecked milestones. The repo contains Monte Carlo modules for scenario loading, registry/discovery, model interfaces, bootstrap/regime models, strategy generation, runner/cache, costs, folds, aggregation, CLI/export support, and tests under `tests/monte_carlo/`. No new issue was filed from this audit because the per-instance stale-status issue already covers it.

### `docs/api.md`

Disposition: `already-correct`

Evidence: the current document no longer claims `POST /analyze`; it lists the implemented FastAPI surface (`GET /health`, `GET /`, `POST /config/patch`, `POST /config/patch/preview`) plus FastAPI framework docs routes. No new issue was filed from this audit because the former phantom-endpoint drift is already corrected.

### `docs/issues/raise_test_coverage_to_89.md`

Disposition: `confirmed-drift`

Evidence: the document says the current temporary coverage threshold is `74%`. Current workflow configuration sets `coverage-min` to `80` in both `.github/workflows/pr-00-gate.yml` and `.github/workflows/ci.yml`.

Follow-up: https://github.com/stranske/Trend_Model_Project/issues/5296

## Validation

- `rg --files-with-matches -g '*.md' '(\[ \]|Pending|TODO|POST /|GET /|Status:)' docs/`
- `rg -n 'app\.add_api_route|@app\.(get|post|put|delete|patch)' src/trend_analysis/api_server tests/test_api_server.py`
- `rg -n 'coverage-min' .github/workflows/pr-00-gate.yml .github/workflows/ci.yml`
