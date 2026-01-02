# Keepalive Status — PR #4113

> **Status:** In progress — settings effectiveness evaluation.

## Progress updates
- Round 1: Confirmed settings extraction now captures model_state assignments and references.
- Round 2: Added mode-specific categorization with context overrides for mode-dependent settings.

## Scope
The Streamlit app exposes 60+ configurable settings, but many have no observable effect on simulation results. Users adjusting settings expect meaningful changes to portfolio construction, but some settings are:
- Not wired through the pipeline
- Only relevant in specific modes/combinations
- Implemented but not producing expected effects
- Display-only with no computational impact

A systematic evaluation framework is needed to:
1. Identify settings that do not produce meaningful output differences
2. Distinguish between "correctly no-op" (mode-specific) vs "broken" settings
3. Ensure new settings are tested before release
4. Provide confidence that user adjustments matter

## Tasks
- [x] Create `scripts/evaluate_settings_effectiveness.py` that:
- [x] Extracts all settings from `streamlit_app/pages/2_Model.py`
- [x] Defines baseline configuration and meaningful test variations
- [x] Runs paired simulations for each setting
- [x] Computes difference metrics (weights, returns, Sharpe, turnover, etc.)
- [x] Applies statistical tests for significance
- [x] Outputs structured results (JSON/CSV)
- [ ] Create `.github/workflows/settings-effectiveness.yml` that:
- [ ] Runs on schedule (weekly) and on-demand
- [x] Executes the evaluation script
- [x] Generates markdown report
- [ ] Opens/updates tracking issue with results
- [ ] Fails CI if effectiveness drops below threshold
- [x] Create `docs/settings/EFFECTIVENESS_REPORT.md` template for results
- [x] Update `scripts/test_settings_wiring.py` to integrate with framework
- [x] Add mode-aware testing (some settings only relevant in specific modes):
- [x] `buy_hold_initial` only relevant when `inclusion_approach=buy_and_hold`
- [x] `rank_pct` only relevant when `inclusion_approach=top_pct`
- [x] `shrinkage_*` only relevant with MVO-based weighting
- [x] `sticky_*` only relevant in multi-period mode

## Acceptance criteria
- [x] Script identifies all settings from UI automatically
- [ ] Each setting tested with at least one meaningful variation
- [x] Results categorized as: EFFECTIVE, MODE_SPECIFIC, NO_EFFECT, ERROR
- [x] Report shows:
- [x] - Overall effectiveness rate (target: >80%)
- [ ] - Per-category breakdown
- [ ] - List of non-effective settings with reasons
- [ ] - Recommendations for each non-effective setting
- [ ] Workflow runs successfully in CI
- [ ] Threshold-based CI failure when effectiveness drops
- [ ] Documentation updated with evaluation methodology
