# Documentation index

Use this index to find the current contributor guides and to understand which overlapping docs remain for historical context.

## Repository Structure

### Core Directories
| Directory | Purpose | Key Files |
| --- | --- | --- |
| `src/` | Main source code | `trend_analysis/` package, `trend_portfolio_app/` |
| `tests/` | Unit and integration tests | See tests/ organization below |
| `config/` | Configuration files | `defaults.yml`, `demo.yml`, `presets/`, `universe/` |
| `scripts/` | Utility and CI scripts | `setup_env.sh`, `run_tests.sh` |
| `docs/` | Documentation | Guides, references, CI docs |
| `analysis/` | Analysis helpers | `cv.py`, `results.py`, `tearsheet.py` |
| `assets/` | Static assets | `screenshots/` for documentation |
| `demo/` | Demo data and outputs | Generated demo datasets |
| `examples/` | Usage examples | Demo scripts, legacy streamlit app |
| `notebooks/` | Jupyter notebooks | `Vol_Adj_Trend_Analysis1.5.TrEx.ipynb` (maintained) |

### Automation & CI
| Directory | Purpose | Key Files |
| --- | --- | --- |
| `.github/workflows/` | GitHub Actions workflows | 36 workflow files (see `WORKFLOW_GUIDE.md`) |
| `.github/actions/` | Custom composite actions | `autofix/`, `build-pr-comment/`, `codex-bootstrap-lite/`, `signature-verify/` |
| `.github/` | GitHub config | `CODEOWNERS`, `agents.json`, `copilot-instructions.md` |

### Archives
| Directory | Purpose | Contents |
| --- | --- | --- |
| `archives/agents/` | Archived Codex task files | 433 files from closed issues (archived 2025-11-30) |
| `archives/github-actions/` | Retired GitHub Actions | 4 unused actions (archived 2025-11-30) |
| `archives/github-config/` | Orphaned GitHub config | `labeler.yml` - no workflow used it (archived 2025-11-30) |
| `archives/analysis/` | Investigation notes | `health44-pr-run-review.md` (archived 2025-11-30) |
| `archives/notebooks/` | Superseded notebooks | Old notebook versions in `2025/` |
| `archives/docs/` | Archived documentation | Historical guides and reports |
| `archives/reports/` | Archived reports | Testing summaries, release notes |
| `docs/archive/` | Documentation archive | See below |

#### Documentation Archive Details (2025-11-30)

**`docs/archive/` structure:**
- `plans/` - 39 completed implementation plans for closed issues
- `ops-issues/` - 5 operations scope docs and smoke test artifacts
- `audits/` - 1 dated audit report
- `ARCHIVE_WORKFLOWS.md` - Retired workflow documentation

**Archived content categories:**
- Issue-specific plans (`issue-XXXX-*.md`) for closed issues: #2523, #2528, #2562, #2566, #2649, #2683, #2688, #2963, #3260, #3261, #3646, #1683, #2812, #2883
- Implementation plans (`*_plan.md`) for completed features: workflow renaming, gate protection, autofix, reusable agents, etc.
- Implementation summaries (`*_SUMMARY.md`, `*_COMPLETE.md`) documenting completed work
- Progress logs (`*_progress.md`, `*-log.md`) for finished tasks
- Dated audit reports (`repository_housekeeping_audit_2025-*.md`)
- Operations issue scopes from `docs/ops/` for closed issues: #2494, #2615, #2820

#### Archive Details (2025-11-30 Cleanup)

**agents/ folder cleanup:**
- Archived 415 `codex-*.md` task files for closed issues
- Archived 18 `ledger-*.md` files for closed issues
- Kept active task files for open issues

**`.github/actions/` cleanup:**
- Archived `apply-autofix/` - superseded by `autofix/`
- Archived `autofix-commit-push/` - superseded by `autofix/`
- Archived `codex-bootstrap/` - superseded by `codex-bootstrap-lite/`
- Archived `update-residual-history/` - no longer referenced
- Kept: `autofix/`, `build-pr-comment/`, `codex-bootstrap-lite/`, `signature-verify/`

**`.github/` root cleanup:**
- Fixed CODEOWNERS stale workflow references
- Archived orphaned `labeler.yml` (no `pr-path-labeler.yml` workflow exists)

**`analysis/` folder cleanup:**
- Archived `health44-pr-run-review.md` investigation notes
- Kept active code: `cv.py`, `results.py`, `tearsheet.py`, `__init__.py`

**`assets/` folder cleanup:**
- Removed empty placeholder PNG files (0 bytes)
- Consolidated placeholder descriptions into `screenshots/README.md`

### Folders Reviewed (No Changes Needed)

**`config/` folder (reviewed 2025-11-30):**
- Well-organized with clear structure
- `defaults.yml` - Master configuration schema
- `demo.yml` - Demo/test configuration
- `presets/` - User presets: `aggressive.yml`, `balanced.yml`, `conservative.yml`, `cash_constrained.yml`
- `universe/` - Universe definitions: `core.yml`, `core_plus_benchmarks.yml`, `managed_futures_min.yml`
- Specialized configs for backtesting, walk-forward analysis, etc.
- Fully documented in `docs/ConfigMap.md`

**`examples/` folder (reviewed 2025-11-30):**
- Contains usage examples with clear README documentation
- Active scripts: `demo_robust_weighting.py`, `demo_turnover_cap.py`, `debug_fund_selection.py`, `integration_example.py`, `portfolio_analysis_report.py`
- `legacy_streamlit_app/` - Historical prototype kept for reference (documented in README_APP.md)

**`notebooks/` folder (reviewed 2025-11-30):**
- Single maintained notebook: `Vol_Adj_Trend_Analysis1.5.TrEx.ipynb`
- Old notebooks already archived to `archives/notebooks/2025/`
- Clear README with maintenance expectations

**`perf/` folder (reviewed 2025-11-30):**
- `perf_baseline.json` - Placeholder for performance regression baseline (not yet generated)
- `runs/` - Transient runtime logs (git-ignored)
- Fixed broken symlink that pointed to non-existent archive path
- Referenced workflow `maint-52-perf-benchmark.yml` does not exist yet

**`reports/` folder (reviewed 2025-11-30):**
- `tearsheet.md` - Generated output from `analysis/tearsheet.py` (active, referenced in README)
- Archived `code_review.md` - one-off analysis from PR #3877

**`scripts/` folder (reviewed 2025-11-30):**
- **74 scripts** organized by category (see `scripts/README.md`)
- CI/Workflow scripts: `ci_*.py`, `ledger_*.py`, `sync_*.py`, `coverage_*.py`
- Core dev scripts: `setup_env.sh`, `run_tests.sh`, `run_streamlit.sh`, `generate_demo.py`
- Validation tiers: `dev_check.sh`, `validate_fast.sh`, `check_branch.sh`
- Performance: `benchmark_performance.py`, `compare_perf.py`
- Utilities: `archive_agents.sh`, `git_hooks.sh`, `docker_smoke.sh`
- Archived `demo_export_fix.py` - one-off fix script

**`streamlit_app/` folder (reviewed 2025-11-30):**
- Primary Streamlit UI for the Trend Portfolio Simulator
- `app.py` - Main entry point with demo button
- `components/` - Reusable UI components (charts, validation, caching)
- `pages/` - Multipage app structure:
  - `1_Data.py` (canonical), `1_Upload.py` (legacy shim)
  - `2_Model.py` (canonical), `2_Configure.py` (legacy shim)
  - `3_Run.py` (canonical), `3_Results.py` (shim → 4_Results)
  - `4_Results.py` (canonical results display)
- Shim files kept for backward test compatibility
- Updated tests to remove references to old `app/streamlit/` path

**`tests/` folder (reorganized 2025-11-30):**
- **3707+ tests** across multiple categories
- **Major reorganization**: Workflow/CI tests separated into `tests/workflows/` for future repo split

| Subfolder | Purpose | Contents |
| --- | --- | --- |
| `tests/workflows/` | **Workflow system tests** | 33 test files, `github_scripts/`, `fixtures/` |
| `tests/app/` | Streamlit app tests | 10 test files |
| `tests/backtesting/` | Backtesting tests | 2 test files |
| `tests/trend_analysis/` | Core analysis tests | 8 test files |
| `tests/scripts/` | Script tests | 6 test files |
| `tests/tools/` | Tool tests | 4 test files |
| `tests/unit/` | Unit tests | `util/` subfolder |
| `tests/smoke/` | Smoke tests | 3 test files |
| `tests/soft_coverage/` | Coverage tracking | 4 test files |
| `tests/golden/` | Golden master tests | 2 test files |
| `tests/fixtures/` | Core test fixtures | `score_frame_2025-06-30.csv` |
| `tests/data/` | Test data files | 6 files |
| `tests/proxy/` | Proxy tests | 1 test file |

**Workflow tests reorganization details:**
- Created `tests/workflows/` to separate CI/automation tests from core tests
- Moved 33 workflow-related test files (`test_workflow_*.py`, `test_autofix_*.py`, `test_ci_*.py`, `test_keepalive_*.py`, `test_agents_*.py`, etc.)
- Moved `tests/github_scripts/` → `tests/workflows/github_scripts/`
- Moved workflow fixtures (`agents_pr_meta/`, `keepalive/`, `keepalive_post_work/`, `orchestrator/`) to `tests/workflows/fixtures/`
- Updated path references in harness JS files and test imports
- Core fixture `score_frame_2025-06-30.csv` remains in `tests/fixtures/`

**`tools/` folder (reviewed 2025-11-30):**
- CI infrastructure utilities imported by workflows and tests
- **14 active files** used by GitHub Actions and test suite

| File | Purpose | Used By |
| --- | --- | --- |
| `post_ci_summary.py` | Build consolidated CI summary | `pr-00-gate.yml`, `maint-46-post-ci.yml` |
| `coverage_guard.py` | Maintain rolling coverage baseline | `maint-coverage-guard.yml` |
| `coverage_trend.py` | Compute coverage trend for CI | `reusable-10-ci-python.yml` |
| `enforce_gate_branch_protection.py` | Ensure branch protection rules | `health-44-gate-branch-protection.yml` |
| `disable_legacy_workflows.py` | Disable retired workflows | `maint-47-disable-legacy-workflows.yml` |
| `resolve_mypy_pin.py` | Resolve mypy version pin | `reusable-10-ci-python.yml` |
| `validate_quarantine_ttl.py` | Validate test quarantine TTL | Tests |
| `agents_index.py` | List agent bootstrap files | Tests |
| `simulate_codex_bootstrap.py` | Simulate Codex bootstrap logic | Tests |
| `simulate_failure_tracker.js` | Test failure tracking | Tests |
| `test_failure_signature.py` | Signature hashing for failures | `health_summarize.py` |
| `strip_output.py` | Strip Jupyter notebook outputs | `pre-commit` hook |
| `pre-commit` | Git pre-commit hook | Git hooks |
| `sanitize_workflows.sh` | Sanitize workflow YAML | Utilities |

**Removed (2025-11-30):**
- `update_workflow_catalog.py` - Generator for stale catalog (catalog itself not tracked)

**`trend_model.egg-info/` (not reviewed - build artifact):**
- Auto-generated by pip during editable install
- Already in `.gitignore` - not tracked
- Safe to delete, regenerates on `pip install -e .`

**`docs/` folder (reviewed 2025-11-30):**
- **62 active documentation files** after archiving completed implementation plans
- Archived 45 files: 39 plans, 5 ops-issues, 1 audit (see `docs/archive/`)

| Category | Contents | Files |
| --- | --- | --- |
| User guides | `UserGuide.md`, `quickstart.md`, `install.md`, `usage.md` | 4 |
| Configuration | `ConfigMap.md`, `config.md`, `PresetStrategies.md` | 3 |
| CLI reference | `CLI.md`, `reference.md`, `api.md` | 3 |
| CI/Workflow | `WORKFLOW_GUIDE.md`, `ci-workflow.md`, `ci_reuse.md`, `checks.md` | 4 |
| Agent automation | `AGENTS_POLICY.md`, `agent-automation.md`, `codex_bootstrap_verification.md` | 3 |
| Validation | `validation-scripts.md`, `efficient-validation.md`, `fast-validation-ecosystem.md` | 3 |
| Dependencies | `DEPENDENCY_ENFORCEMENT.md`, `DEPENDENCY_MANAGEMENT.md`, `DEPENDENCY_SYNC.md`, `DEPENDENCY_WORKFLOW.md` | 4 |
| Development | `code_ownership.md`, `release-process.md`, `pr-iteration-policy.md` | 3 |
| Features | `backtesting_harness.md`, `walk_forward.md`, `Walkforward.md`, `plugin-interface.md`, `metric_cache.md` | 5 |
| Operations | `docs/ops/` - `maintenance-playbook.md`, `ci-status-summary.md`, `codex-bootstrap-facts.md`, `cosmetic-repair.md`, `template-setup.md` | 5 |
| Phase docs | `docs/phase-1/`, `docs/phase-2/` - Design documentation for development phases | 5 |
| Index/Meta | `INDEX.md`, `README.md`, `repository_housekeeping.md` | 3 |

**Notable cleanup decisions:**
- Kept current feature documentation even if it mentions future work
- Kept phase docs as design history (they're properly annotated)
- Kept operations playbooks and runbooks (actively used)
- Archived issue-specific plans when all referenced issues are CLOSED
- Archived implementation summaries (`*_COMPLETE.md`, `*_SUMMARY.md`) - point-in-time snapshots

## Overlapping docs and their scopes
| Document | Audience | Scope/status |
| --- | --- | --- |
| `README.md` | Contributors running the CLI and demos | Quickstart for editable installs and CLI usage; retains legacy actionlint appendix that is being superseded by this index. |
| `README_APP.md` | Streamlit app users and simulation maintainers | Install, layout, and Streamlit/CLI integration notes for the packaged app. |
| `README_DATA.md` | Anyone using bundled datasets | Provenance, intended use, and validation contract for the demo CSVs. |
| `archives/reports/2025-11-22_TESTING_SUMMARY.md` | Test authors and reviewers | Archived upload app test coverage, commands, and portfolio app coverage notes. |
| `archives/docs/2025-11-22_Issues.txt` | Automation/backlog tooling | Historical list of engine/config cleanup tasks kept for parser reference. |
| `ROBUSTNESS_GUIDE.md` | Historical reference | Stub pointing to the archived robustness how-to at `archives/docs/ROBUSTNESS_GUIDE.md`. |
| `docs/README.md` | Documentation entrypoint | Replaces the vendored actionlint README; routes readers here. |

## Canonical navigation

### Onboarding and workflow
- `README.md` for CLI quickstart and demo pipeline usage.
- `docs/WORKFLOW_GUIDE.md` plus `docs/ci/WORKFLOW_SYSTEM.md` for CI/workflow topology and maintenance.

### Testing
- `archives/reports/2025-11-22_TESTING_SUMMARY.md` for the upload and portfolio app testing ledger.
- `scripts/run_tests.sh` for the standard suite entrypoint referenced across CI jobs.

### Robustness
- Archived robustness guide at `archives/docs/ROBUSTNESS_GUIDE.md` until a refreshed version replaces it. Pair it with the weighting options described in `docs/UserGuide.md` when wiring new strategies.

### Data and app flows
- `README_DATA.md` for bundled data constraints and validation helpers.
- `README_APP.md` for Streamlit app packaging and presets.

### Repository hygiene
- `docs/repository_housekeeping.md` for archiving rules, quarterly checklists, and folder ownership.
