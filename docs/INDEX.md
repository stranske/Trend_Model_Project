# Documentation index

Use this index to find the current contributor guides and to understand which overlapping docs remain for historical context.

## Repository Structure

### Core Directories
| Directory | Purpose | Key Files |
| --- | --- | --- |
| `src/` | Main source code | `trend_analysis/` package, `trend_portfolio_app/` |
| `tests/` | Unit and integration tests | pytest test files |
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
