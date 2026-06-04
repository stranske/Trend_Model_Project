# Proposed issues — derived from AUDIT_REPORT.md (independent list)

> Authored from this audit's own findings **before** consulting `review-suggested-issues.md`, to keep the two issue sets independent for comparison. IDs are `A#`. Each maps back to an `AUDIT_REPORT.md` section.

## P1 — correctness, config contract, structure (do first)

| ID | Title | Area | Source |
|----|-------|------|--------|
| **A1** | Make the config contract honest: switch Pydantic models to `extra="forbid"` (or add a declared-vs-consumed key lint) so unknown/inert keys fail loudly | config | §3.3 |
| **A2** | Remove-or-wire the ~19 inert config keys (`metrics.use_continuous`, `metrics.alpha_reference`, `metrics.compute`, `metrics.bootstrap_ci.*`, `export.excel.*`, `export.include_raw_returns/include_vol_adj`, `run.{n_jobs,log_level,log_file,cache_dir,deterministic}`, `preprocessing.steps`) | config | §3.3 |
| **A3** | `data.frequency` silently coerces all data to monthly (√12 regardless) — honor non-monthly end-to-end or hard-reject with a clear error | engine/data | §3.2 |
| **A4** | `data.date_column` ignored on the main CSV load path (only walk-forward/MC honor it) | data | §3.2 |
| **A5** | `data.missing_fill_limit` is a dead alias shadowing the real `data.missing_limit` | config | §3.2 |
| **A6** | Unify the two weighting keys (`portfolio.weighting.name` vs `weighting_scheme`); reject unsupported values loudly; ensure `ScorePropSimple` is reachable from config | portfolio | §3.6 |
| **A7** | Consolidate `backtesting/harness.py` parallel engine (own CostModel/calendar/metrics with **divergent formulas**) — reuse main primitives or retire | dedup/metrics | §2 |
| **A8** | Move the 6 CI/autofix test-fixture modules out of `src/trend_analysis/` (they ship in the wheel) | hygiene/packaging | §1 |
| **A9** | Refactor `multi_period/engine.py::run()` — a single ~3,150-line, ~10-deep function | complexity | §1 |
| **A10** | Add **value-level numeric assertions** to tests (harness + smoke tests check keys, not values) — root cause of undetected wiring/metric bugs | testing | §1 |
| **A11** | Fix the dead `pages/8_Validation.py` (never renders; reads a session key nothing writes) | UX | §4 |
| **A12** | Add a statistical-rigor layer: deflated Sharpe / multiple-testing correction / OOS significance (biggest pre-presentation credibility upgrade) | methodology | §7.2 |

## P2 — economic-wiring nuances

| ID | Title | Area | Source |
|----|-------|------|--------|
| **A13** | `portfolio.cost_model.*` is dead on the main pipeline (engine reads top-level `transaction_cost_bps`/`slippage_bps`) | portfolio/cost | §3.6 |
| **A14** | Single-period path ignores `transaction_cost_bps` (only `monthly_cost`+turnover cap apply) | portfolio/cost | §3.6 |
| **A15** | `regime.threshold=0.0` is degenerate in volatility mode (≈all Risk-Off); validate/reject | regime | §3.4 |
| **A16** | `regime.annualise_volatility` silently rescales the decision boundary by √ppy without rescaling `threshold` | regime | §3.4 |
| **A17** | `regime.min_observations` default mismatch (code 6 vs `defaults.yml` 4) | regime | §3.4 |
| **A18** | `metrics.rf_rate_annual` ignored for ranking unless `rf_override_enabled` (ranking Sharpe uses rf=0) | metrics | §3.4 |
| **A19** | `run.monthly_cost` is an undocumented, material flat cost lever (absent from `defaults.yml`) | cost/docs | §3.4 |
| **A20** | Consolidate the 3 parallelism keys: `run.jobs` canonical, `jobs` deprecated alias, delete dead `run.n_jobs` | config/dedup | §2 |
| **A21** | `sample_split.in_end` boundary convention mismatch (validator=month-start, slicer=month-end) | config/validation | §3.4 |

## P2 — code quality & duplication

| ID | Title | Area | Source |
|----|-------|------|--------|
| **A22** | Consolidate transaction-cost logic fanned across ≥5 implementations into one primitive | dedup | §2 |
| **A23** | Extract the shared CLI skeleton from `run_analysis.py` / `run_multi_analysis.py` | dedup | §2 |
| **A24** | Remove the dead `rebalancing.py` back-compat shim (shadowed by `rebalancing/` package) | dead_code | §2 |
| **A25** | Refactor `cli.main()`/`_handle_mc_command`; narrow ~40 broad `except: pass`; log swallowed config errors (`_compute_score_frame`) | quality | §1 |
| **A26** | `_apply_weight_bounds` may not sum to 1.0 (no warn) + mixes magic `1e-9` with named tolerance | quality/economic | §1 |
| **A27** | Encapsulate module-global mutable cache counters in `rank_selection` | quality | §1 |

## P2 — UX

| ID | Title | Area | Source |
|----|-------|------|--------|
| **A28** | Monte-Carlo page: add numeric filename prefix + connect to the Data→Model→Results state | UX | §4 |
| **A29** | Model page: progressive disclosure + remove internal "Phase N" labels (~4,500-line form) | UX | §4 |
| **A30** | Surface the multi-period weighting silent-fallback (create_weight_engine failure → equal, `fallback_info` not set) | UX/engine | §4 |
| **A31** | Gate developer perf diagnostics out of the production UI (`1_Data.py`) | UX | §4 |
| **A32** | Reconcile the two preset vocabularies (home vs model presets) | UX | §4 |
| **A33** | Make the "Apply selection" commit step clearer | UX | §4 |

## Strategy epics (items 6–7 — larger than a single PR)

| ID | Title | Area | Source |
|----|-------|------|--------|
| **A34** | Survivorship / point-in-time universe handling (as-of membership) | methodology | §7 |
| **A35** | General convex-constraint optimization backend (CVXPY / Riskfolio-Lib) behind the weighting interface | feature | §7.1 |
| **A36** | Factor attribution / returns decomposition (trend purity) | feature | §6 |
| **A37** | Generalize beyond trend (manager-of-managers framing); daily cadence; pluggable cost/regime models; benchmark/peer-index integration | feature/epic | §6, §7.1 |

**Totals:** 12 P1 · 9 P2 (wiring) · 6 P2 (quality/dedup) · 6 P2 (UX) · 4 epics = **37 proposed issues.**
