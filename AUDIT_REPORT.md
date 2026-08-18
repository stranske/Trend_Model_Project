# Trend Model Project — Whole-Repo Audit

**Status:** COMPLETE — all 7 items + executive summary written and cross-reconciled.
**Date:** 2026-06-01/02 · **Branch:** phase-3 · **Auditor:** Claude (Opus 4.8)

**Scope:** Core application — `src/trend_analysis/`, `streamlit_app/`, the `trend` CLI, `config/` — plus supporting `scripts/`, `tools/`, `notebooks/`, `demo/`. Excludes `agents/`, `archives/`, `retired/`, and `.github` automation (owned upstream by `stranske/Workflows`).

**Method:** Independent pass (the pre-existing `review-suggested-issues.md` was deliberately *not* used as input). Findings are produced by fan-out agents reading the code with `file:line` evidence, then adversarially verified before inclusion. Severity: **P0** = wrong numbers / correctness · **P1** = misleading or partial · **P2** = hygiene · **info** = note.

---

## Executive summary

The Trend Model Project is a genuinely **differentiated, well-conceived tool**: an integrated, reproducible, allocator-facing *manager-of-managers* pipeline (score → select top-N → weight → vol-target → multi-period walk-forward → regime overlay → reports) that **no single public tool matches end-to-end on manager return series** (§5). Its core financial arithmetic — metrics, volatility scaling, turnover caps, multi-period turnover-cost — is **correctly wired and economically sensible** (§3).

The dominant weakness is **not the math but the configuration contract.** Across the parameter surface:

- **~19 documented config keys are inert no-ops** — declared in `defaults.yml`/schema but never read by the engine (e.g. `metrics.use_continuous`, the whole `metrics.bootstrap_ci.*` and `export.excel.*` blocks, `run.{n_jobs,log_level,cache_dir,deterministic}`, `preprocessing.steps`). Setting them does nothing, silently (§3.3).
- **A weighting config key silently degrades to equal-weight (YAML/CLI only).** `portfolio.weighting.name` recognizes only `equal`/Bayesian variants; `risk_parity`/`hrp` require a *different* key (`portfolio.weighting_scheme`), and unsupported values fall through to `EqualWeight()` with no error. The shipped `demo.yml` models weighting through exactly the limited key. The **Streamlit GUI is unaffected** (it writes the correct key, §4), but the config-file/CLI path the README leads with is. The advertised "score-proportional" method also appears unreachable from config (§3.6).
- **`data.frequency` silently coerces everything to monthly** (√12 annualisation regardless of input), **`data.date_column` is ignored on the main load path**, **`data.missing_fill_limit` is a dead alias** shadowing the real key, and **`portfolio.cost_model.*` is dead** on the main pipeline (§3.2, §3.6).

**None of these fail loudly** — the highest-leverage single fix is to make the config *honest*: switch the Pydantic models to `extra="forbid"` (or add a declared-vs-consumed key lint) and unify the duplicated weighting/cost/selection keys behind one validated schema that rejects unsupported values.

**Versus the landscape (§5):** strengths are integration, a real manager-ranking front end, a first-class vol-target step, a turnkey regime overlay, a genuine walk-forward engine, bootstrap robustness, and auditability. Gaps vs. mature tools: general convex-constraint optimization (Riskfolio-Lib/PyPortfolioOpt/skfolio), daily-cadence/order-level backtesting (LEAN/vectorbt/cvxportfolio), non-linear cost modeling (cvxportfolio), factor attribution (Venn/Morningstar), cross-validated tuning (skfolio), and a data universe (the SaaS peers).

**Before presenting to colleagues (§7.2),** the questions that will land first are about *statistical credibility*, not features: selection bias / data-snooping when ranking top-N by Sharpe (deflated Sharpe? holdout?), survivorship & backfill bias in the manager universe, look-ahead in selection/regime fitting, gross-vs-net of fees, and whether out-of-sample gains clear the bootstrap bands over an honest baseline (1/N, equal-weight top-N, SG Trend Index).

**Top recommended actions, in order:** (1) make the config honest (`extra="forbid"` + key lint) and unify weighting/cost/selection keys; (2) fix or hard-reject non-monthly `data.frequency` and the `data.date_column` main-path gap; (3) add the statistical-rigor layer (deflated Sharpe / multiple-testing / OOS significance) — the single biggest credibility upgrade; (4) consolidate the parallel implementations flagged in §1–§2 below.

> Items 1 (quality), 2 (duplication), and 4 (UX) are summarized in §1–§4 below; per the agreed priority they were the lower-priority pass.

---

## 1. Code quality

Overall the core is **competently engineered** — typed, configurable, tested, with sensible module boundaries. The issues below are concentrated in a few oversized modules and in error-handling/test-rigor habits. Findings were adversarially verified where flagged P0/P1.

| Finding | Sev | Evidence | Note |
|---|---|---|---|
| **Six CI/autofix *test fixtures* ship inside the production package** | P2 *(verified, ↓ from P1)* | [`_autofix_probe.py`](src/trend_analysis/_autofix_probe.py), `_autofix_violation_case2/3.py`, [`_ci_probe_faults.py`](src/trend_analysis/_ci_probe_faults.py), `automation_multifailure.py`; `pyproject.toml:67` includes `trend_analysis*` with no offsetting exclude | They are self-described autofix fixtures ("not used by production code") yet ship in the wheel as importable `trend_analysis._autofix_*`. `_ci_probe_faults.py` even does runtime work as an import side-effect. **Move to `tests/` fixtures.** |
| **`engine.run()` is a single ~3,150-line function, ~10 levels of nesting** | **P1 *(verified)*** | [multi_period/engine.py:736](src/trend_analysis/multi_period/engine.py:736) (runs to 3886) | The highest-risk maintainability hotspot in the core. A comment at line 3882 ("was incorrectly outside loop causing only last period kept") records a prior correctness bug *caused by* this structure. Extract per-period setup / weighting / cost / result-assembly stages. |
| **The canonical CLI still contains a broad dispatcher and exception boundaries.** | P2 | [cli.py](src/trend/cli.py) | Continue extracting subcommand handlers so `main()` only parses and dispatches. |
| **~40 silent `except: …: pass` / swallow blocks** in `src/trend_analysis` | P2 | [monte_carlo/runner.py:1673](src/trend_analysis/monte_carlo/runner.py:1673), runner.py:1770, cli.py:366 | Broad catches in numeric paths mask real defects (e.g. `AttributeError`). Narrow the types and log at debug. |
| **`_apply_weight_bounds` can return weights not summing to 1.0**; mixes a magic `1e-9` with the named tolerance | P2 | [multi_period/engine.py:342](src/trend_analysis/multi_period/engine.py:342) | If all donors/receivers are saturated, residual is silently left; no warn. Mildly *economic* (weights should sum to 1). |
| **Module-global mutable cache counters** (`_SELECTOR_CACHE_HITS/MISSES`) | P2 | [core/rank_selection.py:339](src/trend_analysis/core/rank_selection.py:339) | Test-pollution / thread-safety hazard; derive from `_WINDOW_METRIC_CACHE.stats()` instead. |
| **`_compute_score_frame` swallows config errors → empty frame, silently** | info *(verified ↓ from P2)* | [monte_carlo/runner.py:1196](src/trend_analysis/monte_carlo/runner.py:1196) | At least log the exception before returning empty. |
| **Tests assert keys exist, not numeric values** | P2 | [tests/backtesting/test_harness.py:133](tests/backtesting/test_harness.py:133), [tests/smoke/test_pipeline_smoke.py:30](tests/smoke/test_pipeline_smoke.py:30) | **This is *why* the §3 wiring issues and divergent metric formulas go unnoticed** — value-level assertions on deterministic fixtures would catch regressions. The single most valuable test-quality upgrade. |

## 2. Duplicative / repetitive code

The codebase has accumulated several **parallel implementations of the same concept**. The dedup agent verified which are genuine redundancy vs. justified separation.

**Genuine consolidation targets (highest value first):**

1. **`backtesting/harness.py` is a second, parallel backtest engine** — high effort, high value. It reimplements `CostModel` ([harness.py:31](src/trend_analysis/backtesting/harness.py:31)), `_rebalance_calendar` (harness.py:533), `_normalise_frequency`/`_infer_periods_per_year` (harness.py:540/555), and its own metrics — **with formulas that diverge from the main pipeline.** This is the structural root of the kind of metric inconsistency that's easy to ship (two code paths computing "the same" Sharpe/Sortino/cost differently). Decide whether the harness is still needed; if so, have it call the main `metrics/`, `cost`, and calendar primitives rather than its own.
2. **Transaction-cost logic fanned across ≥5 implementations** — medium. `CostModel.apply` (harness), [metrics/turnover.py:39](src/trend_analysis/metrics/turnover.py:39) `turnover_cost`, [rebalancing/strategies.py:199](src/trend_analysis/rebalancing/strategies.py:199) `_calculate_cost`, [monte_carlo/costs.py:172](src/trend_analysis/monte_carlo/costs.py:172), and the inline `period_cost` in the engine. Define the linear turnover-cost primitive **once** and call it everywhere. (Corroborates the §3.6 cost-fragmentation finding.)
3. **Two parallel weighting config keys** — medium. `portfolio.weighting.name` resolver ([engine.py:1603](src/trend_analysis/multi_period/engine.py:1603)) vs `portfolio.weighting_scheme` resolver (engine.py:1625), chosen between in `_compute_weights`. Unify under one user-facing key whose value space spans *both* the score-based schemes and the risk-based engine names. (See §3.6.)
4. **`run_analysis.py` vs `run_multi_analysis.py`** share a CLI/export skeleton — medium. Extract the shared argparser/logging/config-load/out-dir boilerplate into one helper. (`pipeline_entrypoints.py` is correctly centralized — leave it.)
5. **Three parallelism keys** — low. `run.jobs` (live), top-level `jobs` (legacy alias), `run.n_jobs` (**dead**). Keep `run.jobs` canonical, treat `jobs` as a documented deprecated alias resolved in one place, delete `run.n_jobs`.

**Verified *not* duplication (don't merge):**

- **`config/model.py` vs `config/models.py`** — *justified layer separation* (strict Pydantic validator vs. runtime-dict `Config` factory with a fallback-import contract). Verifier downgraded to **info**; the only cleanup is clearer naming to stop the two from being mistaken for each other. *(This corrects an earlier impression in §3.6.)*
- **`rebalancing.py` (module) vs `rebalancing/` (package)** — an intentional back-compat shim. The actionable item is that the package shadows the module on disk, so `rebalancing.py` is almost certainly **dead** and can be removed.

## 3. Functional & economic correctness — parameter wiring

**Method.** Every user-facing config parameter was inventoried (definition site + every consumption site found via ripgrep), then judged on two axes — *wiring* (does changing it actually change results?) and *economic sensibility* (does the output move in the direction/magnitude a knowledgeable PM would expect?). High-impact findings were then handed to an independent agent told to **refute** them; only findings that survived (or were severity-corrected) are reported. The `portfolio.*` domain (selection/weighting/costs) is being re-run after a stall and will be folded in below.

### 3.1 Scorecard (data / preprocessing / vol_adjust / sample_split / regime / benchmarks / metrics / export / run)

| Domain | Params mapped | Confirmed correct | Dead / not-wired | Economic / wiring concern |
|---|---|---|---|---|
| data·preprocessing·vol_adjust·sample_split | 16 | 10 | 2 | 5 |
| regime·benchmarks·metrics·export·run | 41 | ~17 | ~17 | 7 |
| portfolio (selection·weighting·costs·rebalance) | ~14 | ~6 | ~2 (`cost_model.*`) | 4 (weighting bifurcation, cost path, single-period costs) |

The single biggest theme: **a large fraction of the advertised config surface is inert.** ~19 documented keys (in `config/defaults.yml`, `demo.yml`, or the generated JSON schema) are *never read by any engine code*. A user who sets them gets silent no-ops. This is the most important correctness/UX risk in the parameter layer — the config file advertises capabilities the engine does not have.

### 3.2 Headline findings (P1, survived adversarial verification)

- **`data.frequency` silently coerces everything to monthly** — *P1, confirmed.* `frequency: D` or `W` only changes an intermediate calendar-alignment cadence; `_prepare_input_data` then unconditionally resamples to month-end (`ME`), and `periods_per_year` is hardcoded to 12 on both the single-period and multi-period paths. So a PM choosing daily expecting √252 annualisation and daily-cadence trend/vol windows actually gets monthly bars with √12 — windows like `window=63` silently change economic meaning. Evidence: [preprocessing.py:450](src/trend_analysis/stages/preprocessing.py:450), [util/frequency.py:110](src/trend_analysis/util/frequency.py:110), [multi_period/engine.py:1230](src/trend_analysis/multi_period/engine.py:1230). **Fix options:** either honor non-monthly cadence end-to-end, or reject non-`M` frequency at validation with a clear message.
- **`data.missing_fill_limit` is a dead alias that shadows the real key** — *P1, confirmed.* It is in the schema description ("Maximum consecutive periods to forward-fill") and in `demo.yml`, but `DataSettings` has no such field (`extra='ignore'` drops it) — the live key is `data.missing_limit`. Copy-paste demo users who tune `missing_fill_limit` get nothing. Evidence: [schema_generator.py:106](src/trend_analysis/config/schema_generator.py:106), [model.py:195](src/trend_analysis/config/model.py:195), [model.py:199](src/trend_analysis/config/model.py:199).
- **`portfolio.weighting.name` silently degrades to equal-weight for YAML/CLI runs** — *P1 (see §3.6).* This config key recognizes only `equal` and Bayesian variants; `risk_parity`/`hrp` require the *separate* `portfolio.weighting_scheme` key, and unsupported values fall through to `EqualWeight()` with no error. The shipped `demo.yml` models weighting through exactly this limited key. **The Streamlit GUI is unaffected** — it writes the correct `weighting_scheme` key (verified in §4).

### 3.3 Inert / dead config keys (silent no-ops)

These are declared in config/schema but have **zero consumers** in the engine. Each is a UX trap (the app implies a capability it lacks). Verified examples (`run.n_jobs` and `metrics.use_continuous` were independently confirmed dead, severity-corrected to P2):

| Key | Notes |
|---|---|
| `metrics.use_continuous` | geometric-vs-log return toggle — no consumer ([defaults.yml:153](config/defaults.yml:153)) |
| `metrics.alpha_reference` | alpha/beta benchmark ticker — no consumer |
| `metrics.compute` | a *second* metric list (Title_Case) that is dead; only `metrics.registry` is live, and the two disagree |
| `metrics.bootstrap_ci.{enabled,n_iter,ci_level}` | CI is actually driven by `portfolio.ci_level`; this block is inert |
| `export.excel.{autofit_columns,number_format,conditional_bands.*}` | declared but `export_to_excel` never reads them |
| `export.{include_raw_returns,include_vol_adj,include_figures}` | inclusion not gated on these |
| `run.{log_level,log_file}` | logging always uses `INFO` and a generated path |
| `run.n_jobs` | dead; **duplicates** the live top-level `jobs` / `run.jobs` |
| `run.cache_dir` | cache dir comes from `$TREND_ROLLING_CACHE` / `~/.cache`, not this |
| `run.deterministic` | determinism is unconditional via `seed`; the toggle does nothing |
| `preprocessing.steps` | implies a configurable pipeline; sequence is hardcoded |

**Recommended fix pattern:** switch the relevant Pydantic models from `extra='ignore'` to `extra='forbid'` (or add a config-lint pass that diffs declared keys against a registry of consumed keys) so unknown/inert keys fail loudly instead of silently. This single change would have caught most of the above.

### 3.4 Economic-direction & consistency concerns (P2)

- **`vol_adjust.target_vol` leverage is invisible in the weights** — returns are scaled correctly (higher target_vol → higher realised vol/return, the right direction), but the *weights* are renormalised to sum to 1 ([risk.py:291](src/trend_analysis/risk.py:291)), so a PM inspecting the holdings table sees an identical inverse-vol tilt for `target_vol=0.05` and `0.50`. Leverage > 100% gross is never surfaced as exposure. Presentation hazard, not a numeric error.
- **`regime.threshold=0.0` is degenerate in volatility mode** — `signal = threshold − vol` with non-negative vol means ~every period is classified Risk-Off under shipped defaults; the volatility regime method only works if the user sets a positive vol target, and nothing validates this. Evidence: [regimes.py:213](src/trend_analysis/regimes.py:213). (Verifier corrected to P2 since the default `method` is `rolling_return`, not `volatility`.)
- **`regime.annualise_volatility`** silently rescales the regime decision boundary by √ppy without rescaling `threshold` — flipping it changes the entire Risk-On/Risk-Off split for a fixed threshold.
- **`regime.min_observations` default mismatch** — code defaults to 6, `defaults.yml` ships 4 ([regimes.py:83](src/trend_analysis/regimes.py:83) vs [defaults.yml:139](config/defaults.yml:139)).
- **`metrics.rf_rate_annual` is ignored for ranking** unless `rf_override_enabled=true`; with shipped defaults, fund-selection Sharpe uses rf=0 even though `rf_rate_annual: 0.02` is configured — can subtly change *which funds are selected* vs. user expectation. Evidence: [api.py:462](src/trend_analysis/api.py:462).
- **`data.frequency` rf de-annualisation mismatch** (*P2, confirmed*) — when `rf_override_enabled` and frequency is non-monthly, the periodic rf is computed with 52/252 but subtracted from the monthly return series, understating the rf deduction (overstating Sharpe). [api.py:467](src/trend_analysis/api.py:467).
- **`run.monthly_cost` is an undocumented, material cost lever** — widely consumed (subtracted flat from every per-period return: [portfolio.py:571](src/trend_analysis/stages/portfolio.py:571)) but absent from `defaults.yml`. Direction is sensible (higher → lower net return) but it is a flat fee unrelated to turnover/exposure, and being undocumented it hides a meaningful knob.
- **`sample_split.in_end` boundary convention mismatch** — the ordering validator parses `'2022-12'` as month-*start*; the window slicer resolves it to month-*end* with an inclusive mask. Safe for distinct adjacent months but fragile (two conventions for one field).
- **Parallelism config is fragmented** across `jobs` (top-level, live), `run.jobs` (live only on the Monte-Carlo CLI path), and `run.n_jobs` (dead) — should be one coherent knob.

### 3.5 What is correctly wired (positive notes)

`vol_adjust.{enabled,target_vol}` return-scaling, `data.{missing_policy,missing_limit,risk_free_column,allow_risk_free_fallback,csv_path}`, `preprocessing.missing_data.{policy,limit}`, `sample_split.in_start`, `regime.{enabled,risk_off_target_vol_multiplier,risk_off_fund_count_multiplier}`, `benchmarks`, `metrics.{registry,rf_override_enabled}`, `export.{directory,formats,disable_narrative_generation}`, and `run.seed` were all confirmed correctly wired with sensible economic behavior.

### 3.6 Portfolio domain (selection / weighting / costs / rebalancing)

_(Audited directly after two workflow attempts on this slice failed — once a stall, once a structured-output miss — so further speculative agent spend wasn't warranted.)_

**Headline — the weighting parameter is the most confusingly-wired knob in the app.** Weighting is controlled by **two different config keys with two different resolvers and two different vocabularies**:

1. `portfolio.weighting.name` (the dict form used in `demo.yml`) is resolved at [multi_period/engine.py:1608](src/trend_analysis/multi_period/engine.py:1608). It recognizes **only** `equal`/`ew`, the Bayesian variants (`score_prop_bayes`/`bayes`/`score_bayes`, `adaptive_bayes`/`adaptive`) — and **everything else silently falls through to `EqualWeight()`** ([engine.py:1622](src/trend_analysis/multi_period/engine.py:1622)).
2. `portfolio.weighting_scheme` (a separate string key) is resolved at [engine.py:1627](src/trend_analysis/multi_period/engine.py:1627) and handles the risk-based engines: `risk_parity`, `hrp`, `erc`, `robust_mv`, `robust_risk_parity`. A code comment states it *"overrides the legacy `portfolio.weighting` dict config."*

**Scope of the impact (reconciled with the §4 UI review).** The Streamlit GUI is **not** affected: it writes the method to `portfolio.weighting_scheme`, which both engines read correctly. The trap bites **YAML/CLI users** who configure weighting through the dict form the demo models. So this is a config-file footgun, not a broken GUI — important for severity, but still serious for the scripted/CLI workflow the README leads with.

Consequences (P1 for YAML/CLI users; GUI unaffected):

- **`portfolio.weighting.name: risk_parity` silently produces equal weight** (for a config-file/CLI run). To actually get risk parity you must set the *other* key, `portfolio.weighting_scheme: risk_parity`. The shipped `demo.yml` only sets `portfolio.weighting: {name: equal}` — so the natural place a config-file user would change the method is the one that doesn't work for risk-based schemes, and it fails silently rather than erroring.
- **The advertised "score-proportional" method appears unreachable from config.** `ScorePropSimple` exists at [weighting.py:31](src/trend_analysis/weighting.py:31), but neither resolver maps a config name to it (the `weighting.name` branch only wires the *Bayesian* score variant; `weighting_scheme` only wires risk engines). The README advertises "score-proportional" weighting, but I could find no config string that selects plain `ScorePropSimple`. *(Worth a confirmation pass, but the two resolvers above are the only ones I found.)*
- This directly answers item 3 for the most important parameter: **changing `portfolio.weighting.name` does NOT reliably change the weighting method, and does not fail loudly when the requested method is unsupported.**

**Transaction costs — fragmented across three implementations, and `cost_model.*` is dead on the main path.**

- The multi-period engine charges `period_cost = period_turnover × ((tc_bps + slippage_bps) / 10000)` ([engine.py:3640](src/trend_analysis/multi_period/engine.py:3640)) — turnover-proportional and economically correct. ✓
- **But it reads top-level `portfolio.transaction_cost_bps` and `portfolio.slippage_bps`** ([engine.py:1684-1685](src/trend_analysis/multi_period/engine.py:1684)), **not** the validated `portfolio.cost_model.{bps_per_trade, slippage_bps}`. `cost_model` has full validation in both `config/model.py` and `config/models.py` and is in the JSON schema, yet **`cost_model` is never read by the main pipeline** (`rg cost_model` over engine.py / stages/portfolio.py / api.py is empty) — it is consumed only by the separate `backtesting/harness.py`. So a user who configures `portfolio.cost_model` for a normal `trend run` gets no cost effect (P1), and the engine's `portfolio.slippage_bps` is an undocumented top-level key not present in the schema (P2).
- **Single-period analysis ignores transaction costs entirely.** `stages/portfolio.py` applies the flat `monthly_cost` and the `max_turnover` cap but never `transaction_cost_bps` (no turnover-based cost on the single-period path). So `transaction_cost_bps` is a silent no-op unless you run multi-period. (P2 — arguably defensible since single-period has one allocation, but it is surprising and undocumented.)

**Correctly wired (positive notes).**

- `portfolio.max_turnover` is honored on both paths (turnover cap → trade clipping), with **regime-aware caps** and a `lambda_tc` turnover-damping penalty in multi-period ([engine.py:201](src/trend_analysis/multi_period/engine.py:201), [engine.py:459](src/trend_analysis/multi_period/engine.py:459)). ✓
- `portfolio.selection_mode` and the selector plugin registry (`rank`, `zscore`) are wired ([api.py:533](src/trend_analysis/api.py:533), [selector.py:9](src/trend_analysis/selector.py:9)). ✓
- The drop-on-load bug (selection/weighting/constraints lost because they weren't declared model fields) is **fixed** via `extra="allow"` with an explanatory comment ([config/model.py:370-374](src/trend_analysis/config/model.py:370)). ✓
- `risk_parity` / `hrp` / `erc` / robust engines are real implementations under `weights/` and are reachable via `weighting_scheme`. ✓

**Secondary smell (feeds §1/§2).** Selection is *also* doubly-specified: `demo.yml` carries both a `portfolio.rank` block and a `portfolio.selector` block expressing the same intent (top-5 by Sharpe). The relationship between them is not obvious from config. Combined with the two weighting keys and the parallel `backtesting/harness.py` metrics/cost path (§2), the portfolio layer has accumulated overlapping mechanisms worth consolidating. *(Note: the `config/model.py` vs `config/models.py` split, which initially looked like another duplicate, was verified in §2 to be **justified** layer separation — strict validator vs. runtime-dict factory — not a merge target.)*

### 3.7 Net assessment for item 3

- **Wiring is correct for the *core arithmetic*** — metrics, vol scaling, turnover caps, and multi-period turnover-cost all move output in economically sensible directions.
- **The dominant problem is the *configuration contract*:** ~19 documented keys are inert no-ops, the headline `weighting.name` knob silently degrades to equal-weight for unsupported values, an advertised method (score-proportional) appears unreachable, `cost_model` is dead on the main path, and several knobs (`frequency`, `date_column`, transaction costs) behave differently than a PM would reasonably expect. None of these fail loudly.
- **Highest-value single fix:** make the config *honest* — switch the Pydantic models to `extra="forbid"` (or add a lint that diffs declared-vs-consumed keys), and unify the weighting/cost/selection keys behind one validated schema that rejects unsupported values. That converts a class of silent economic-misconfiguration risks into loud, fail-fast errors.


## 4. Design & ease of use (UX)

_Static review of the Streamlit app (the live app was **not** launched — see coverage note in the appendix). Pages reviewed: `app.py`, `state.py`, `config_bridge.py`, and `pages/{1_Data, 2_Model, 3_Results, 4_Help, 8_Validation, monte_carlo}.py` plus key components._

**Overall verdict.** A knowledgeable allocator can broadly follow the intended **Data → Model → Results** flow, and those three core pages are visually coherent (consistent emoji-prefixed subheaders, sensible column layouts, `st.metric` summary tiles, tabbed Results). Defaults are mostly reasonable and the **Help page is genuinely good**. But two pages are effectively broken as shipped, the information architecture has a real disconnect at the edges, and the Model page is overwhelming for a first-time user.

**Notable correction to §3.6 (verified in the UI code):** the feared catastrophic "silent equal-weight" trap does **not** manifest in the app. The Model page stores the choice in `model_state['weighting_scheme']`, `analysis_runner` writes it to `portfolio.weighting_scheme` ([analysis_runner.py:337](streamlit_app/components/analysis_runner.py:337)), and **both** the single-period ([api.py:497](src/trend_analysis/api.py:497)) and multi-period ([engine.py:1627](src/trend_analysis/multi_period/engine.py:1627)) engines read that key and dispatch `risk_parity`/`hrp`/`erc`/`robust_*` correctly through the plugin registry. So the GUI route is sound; the `weighting.name` equal-only branch only bites **YAML/CLI** users (and the threshold-hold policy path the UI never enables).

| Page | Issue | Sev |
|---|---|---|
| **`8_Validation.py`** | **The page never renders** — unlike every other page it doesn't call its render function (no unconditional call / `_should_auto_render()`); and even if it did, it reads uploaded data from the wrong session key (`st.session_state['app_data']['returns']`, lines 543-548) which no other module writes, so it would `st.stop()`. A **dead/broken page** shipped in the nav. | **P1** |
| **`monte_carlo.py`** | (a) **IA disconnect** — driven by a separate scenario registry (`list_scenarios`/`load_scenario`) rather than the Data→Model→Results state, so it doesn't consume the model the user just built. (b) **Filename lacks the numeric prefix** every other page uses, so Streamlit orders/labels it inconsistently in the sidebar. | P2 |
| **`2_Model.py`** | A single **~4,500-line form** exposing dozens of parameters, with internal dev labels leaking to users (`"Fund Holding Rules (Phase 3)"`, `"Hard thresholds (Phase 13)"`). Overwhelming; needs progressive disclosure (basic/advanced) and user-facing language. | P2 |
| **`2_Model.py`** | **Residual weighting trap:** in multi-period mode, if `create_weight_engine` raises during construction the engine silently sets `use_risk_weighting=False` and falls back to equal weight ([engine.py:1651](src/trend_analysis/multi_period/engine.py:1651), "best-effort only") **without populating `fallback_info`**, so the Results-page fallback banner (3_Results.py:2044) won't fire — the allocator could believe they got risk-parity when they got equal. Surface this fallback. | P2 |
| **`1_Data.py`** | **Developer diagnostics rendered in the production UI** on every run — an always-visible perf caption (`"Perf: total Xms | render Yms | seed Zms …"`, lines 736-750). Gate behind a debug flag. | P2 |
| **`app.py`** | **Two parallel preset vocabularies** — the home "Strategy Preset" selector (from `demo_runner.list_presets()`) vs. the model presets elsewhere — create confusion about which presets are authoritative. | P2 |
| **`1_Data.py`** | **"Apply selection" is an easy-to-miss step** — toggling fund checkboxes updates a live count but only commits to `analysis_fund_columns` on a separate action; users can think they've selected funds when they haven't. | P2 |

**Strengths:** the **Results page** information architecture is strong (clear gating sequence: error-if-no-data → error-if-no-model → prompt-to-run → success banner with period count/date range, then tabbed Summary/Period/Viz/Fund/Export/Compare). The **Help page** with an in-page table of contents is a real asset (minor risk: its anchor links depend on auto-generated header slugs).

**Net for item 4:** intuitive enough for a knowledgeable allocator on the happy path, visually coherent on the three core pages, but let down by (1) a dead Validation page, (2) a disconnected Monte-Carlo page, and (3) an over-dense Model page that exposes internal "Phase N" scaffolding. Fixing those three would materially raise first-use intuitiveness.

## 5. Effectiveness of the approach vs. public alternatives

_Based on primary-source web research across 28 comparable tools in five categories (raw research + verified-vs-codebase synthesis stashed in [reports/audit/landscape_raw.json](reports/audit/landscape_raw.json)). Point-in-time facts (star counts, versions, maintenance status) are flagged as lower-confidence at the end._

### 5.1 Where this app sits

This app is an **integrated, allocator-facing manager-of-managers pipeline**: rank managers by a configurable risk-adjusted score → select top-N → weight → optionally vol-target the book → single/multi-period walk-forward backtest → risk-on/risk-off regime overlay → Excel/CSV/JSON/HTML/PDF reports, built specifically for **monthly trend-following / managed-futures manager returns**. **No single tool in the landscape spans this whole arc on manager returns.** It occupies the gap between two mature clusters:

- **Open-source quant libraries/engines** (powerful *construction* or *backtest*, but instrument/price-centric and missing the manager-ranking front end).
- **Institutional manager-research SaaS** (rich data + factor attribution, but closed and non-reproducible — you cannot tune or audit the score/selection/cost logic).

### 5.2 Closest analogues, by axis

| Tool | Category | Closest on… | Relative to this app |
|---|---|---|---|
| **bt** (pmorissette) | OSS backtest | **Architecture** — `SelectN → WeighEqually/WeighInvVol/WeighERC → TargetVol → RunMonthly` maps ~1:1 onto this app | This app adds: manager *scoring*, Bayesian weighting, a regime overlay, a first-class walk-forward engine, allocator reports. bt adds: nothing this app lacks except a commission callback. |
| **skfolio** | OSS portfolio opt | **Pipeline shape** — pre-selection → weighting → leakage-safe `WalkForward`/purged CV; sklearn-native | skfolio has far deeper CV/hyperparameter tuning and a general convex constraint layer; this app has the manager front end + vol-target + regimes it lacks. |
| **Riskfolio-Lib / PyPortfolioOpt** | OSS portfolio opt | **Weighting depth** — 20+ risk measures, full MV/Black-Litterman frontier, turnover/cardinality/tracking-error constraints | This app's weighting is mostly closed-form/heuristic with only min/max bounds — shallow by comparison. |
| **cvxportfolio** (Boyd) | OSS portfolio opt | **Multi-period + cost realism** — market-impact and holding-cost terms in a convex multi-period objective | This app's cost model is linear bps-on-turnover; cvxportfolio is the gold standard for cost-aware multi-period. |
| **vectorbt** | OSS backtest | **Robustness throughput** — thousands of parameter combos + walk-forward CV, fast | This app can't sweep/tune at that scale; but it has the construction semantics vectorbt deliberately omits. |
| **QuantConnect LEAN** | OSS engine | **Construction menu + execution realism** — EqualWeight/MeanVariance/BlackLitterman/RiskParity models, fee+slippage models, daily/tick, live trading | Institutional-heavyweight; overkill for monthly manager selection and built for tradeable securities, not manager returns. |
| **pysystemtrade / Clenow / PyTrendFollow** | Trend/CTA | **Domain** — systematic futures trend with vol targeting | These build trend signals *from price series*; this app instead *selects among managers* who already run such systems — a complementary layer up. |
| **SG Trend Index** | Trend/CTA | **Purpose** — a real rules-based, equal-weight, top-N pool of trend CTAs | This app *generalizes* the SG Trend construction to scored ranking + richer weighting; SG Trend is the natural benchmark to regress selected portfolios against. |
| **Venn / Morningstar Direct / eVestment / PivotalPath / Portfolio Visualizer** | Allocator SaaS | **Evaluate-and-rank front half** + curated universes + factor attribution | None run a *reproducible* rule-based walk-forward selection+weighting+vol-target simulation; this app's auditability and code-control are its edge over them. |
| **Two Sigma GMM regime (via Venn)** | Regime | **Regime modeling** — probabilistic multi-state GMM on a factor lens | This app's regime overlay is a coarse binary rolling-return/vol threshold by comparison. |

### 5.3 Genuine strengths of this approach

1. **End-to-end integration on manager return series** — the only tool here that bundles score-ranking → top-N → weighting → vol-target → multi-period walk-forward → allocator reports in one pipeline. Every alternative covers only a slice.
2. **A real manager-ranking/selection front end** keyed to a configurable composite risk-adjusted score — the portfolio-optimization libraries and the trend/CTA tools have *no* first-class ranking/screening step.
3. **A genuinely competitive weighting *implementation* menu** — `EqualWeight`, `ScorePropSimple`, `ScorePropBayesian`, `AdaptiveBayesWeighting`, `RiskParity`, `ERC`, `HRP`, `RobustMeanVariance`, `RobustRiskParity`. This reaches into skfolio/Riskfolio-Lib/PyPortfolioOpt territory. **⚠️ Caveat (from §3.6): the implementations exist, but the config can't reliably *reach* several of them** — the headline strength is undercut by the wiring bug.
4. **A first-class, named book-level vol-target step** — bt's `TargetVol` is the only OSS analog; PyPortfolioOpt/Riskfolio-Lib/skfolio/cvxportfolio reach vol targets only indirectly via constraints.
5. **A turnkey risk-on/risk-off regime overlay** — DIY in every OSS engine here.
6. **A genuine multi-period walk-forward engine** with in/out-of-sample period generation — bt/backtrader/zipline/PyPortfolioOpt/Riskfolio-Lib have none.
7. **Statistical robustness present** — circular block-bootstrap equity-curve bands + a reporting CI level — exceeds the ad-hoc robustness of bt/backtrader and the optimizer libraries.
8. **Reproducible, auditable, config-driven** (with the recent run-envelope/`run_contract` JSON and deterministic identity-map work) — a clear edge over the closed SaaS peers where score/selection/cost logic can't be tuned or audited.

### 5.4 Weaknesses & gaps vs. mature alternatives

1. **Shallow constraint-based optimization** — only min/max weight bounds; no general convex constraint layer (sector/group bounds, cardinality, tracking-error, explicit turnover budget) as in Riskfolio-Lib / PyPortfolioOpt / skfolio.
2. **Monthly-only cadence** — no daily/intrabar/order-level simulation (LEAN, backtrader, zipline, vectorbt, cvxportfolio all do). Compounded by the §3 finding that `data.frequency` silently coerces to monthly regardless.
3. **Linear cost model only** — no market-impact, volume-share slippage, holding/financing costs, or per-asset-class fees (cvxportfolio, LEAN, backtrader, pysystemtrade have these).
4. **No factor attribution / returns decomposition** — can't explain *why* a manager earns its returns or gauge trend purity/replicability (Venn's Two Sigma Factor Lens, Morningstar style analysis, Finominal).
5. **Single rolling re-fit, not cross-validated tuning** — no purged/combinatorial CV or GridSearch-style hyperparameter selection (skfolio) and no high-throughput sweeps (vectorbt).
6. **No efficient-frontier MV/Black-Litterman solver** of the depth in PyPortfolioOpt / Riskfolio-Lib / cvxportfolio.
7. **Coarse binary regime model** vs. probabilistic multi-state GMM/HMM approaches.
8. **No data universe, no live/broker path, no native CTA peer indices** — you must supply your own manager returns; SaaS peers ship thousands of curated strategies + qualitative DD overlays.
9. **Maturity/adoption gap** — a single-purpose project vs. battle-tested tools (LEAN, Riskfolio-Lib, PyPortfolioOpt, pysystemtrade, skfolio) and institutional SaaS with analyst overlays.

### 5.5 Caveats (lower-confidence research claims)

Point-in-time figures (star counts; bt v1.2.0; vectorbt OSS v1.0.0; skfolio v0.20.1; etc.) and maintenance-status calls (backtrader "legacy"; PyTrendFollow "unmaintained since 2018"; cvxportfolio "~11 months stale") should be re-verified before publishing a head-to-head. Proprietary methodology details (PivotalPath PQP; Two Sigma's four-regime GMM labels) come partly from trade press. The "bt lacks native Bayesian weighting / WFO" claim — used to credit this app — is consistent with bt's documented Algo set but warrants a docs check before a public comparison.

## 6. Missed opportunities to extend to adjacent problems

The pipeline is **strategy-agnostic on returns** — it operates on a matrix of periodic manager returns and a score. That generality is mostly latent; a handful of modest extensions would let the same engine solve closely related allocator problems:

1. **Generalize beyond trend/managed-futures to any manager universe.** Nothing in the ranking/selection/weighting/regime machinery is trend-specific. Re-framing the docs and presets around "manager-of-managers for *any* strategy bucket" (equity L/S, global macro, multi-strat, credit) is almost free and multiplies the addressable use cases. The only trend-specific assumptions are in the *defaults* (e.g. SPX regime proxy), not the engine.
2. **Asset-class / index allocation (TAA/SAA).** The same select→weight→vol-target→walk-forward flow applies to ETF or asset-class index returns. A "strategic/tactical asset allocation" preset would reach a much larger audience (the Portfolio Visualizer crowd) with no engine change.
3. **Benchmark replication / tracking.** Given the SG Trend Index analog, add a "track/replicate this index" objective — select+weight managers to minimize tracking error to a target index. This is a natural, high-value adjacent problem and a direct competitor to commercial CTA-replication analytics (Finominal).
4. **Fee-structure–aware manager selection.** Manager selection lives or dies on *net*-of-fee returns. Today cost is a flat `monthly_cost` + linear turnover bps. Modeling management + performance fees with hurdles/high-water-marks (a manager-specific concern absent from every generic backtester) would be a differentiator, not just a gap-filler.
5. **Survivorship/point-in-time universe handling** → unlocks credible historical studies and "as-of" rebalancing (see §7 — this is also a correctness concern).
6. **Factor attribution as a selection input.** Adding a returns-based factor regression (trend factor, equity beta, carry) would let the tool rank on *trend purity / alpha* rather than raw Sharpe, closing the biggest analytical gap vs. Venn/Morningstar and improving selection quality.
7. **Scenario / stress testing tied to the regime engine.** The regime overlay already partitions history; exposing "show me performance conditioned on regime X / a user-defined stress window" is a small step from existing code and is exactly what allocators ask for.
8. **Parameter-sensitivity / robustness surface.** The bootstrap exists; a built-in sweep over `top_n` / `score_by` / `lookback` / rebalance cadence (à la vectorbt) would turn one-off backtests into stability analysis — and would surface the §3 wiring traps loudly.

## 7. Generalizing the model + questions to ask before presenting to colleagues

### 7.1 What it would take to generalize the model

In rough order of leverage:

1. **Honor `data.frequency` end-to-end (precondition for everything else).** The hardcoded monthly resample + `periods_per_year=12` (see §3.2) is the single biggest generalization blocker — daily/weekly manager or asset data is silently downgraded. Thread a real `periods_per_year` from the resolved frequency through `preprocessing`, annualisation, the trend/vol windows, and the multi-period scheduler.
2. **Make the config contract honest and unified (the §3 fixes).** Switch Pydantic models to `extra="forbid"` (or add a declared-vs-consumed key lint), collapse the two weighting keys (`weighting.name` vs `weighting_scheme`) and the two selection blocks (`rank` vs `selector`) into one validated schema, and wire `cost_model.*`. Until config is trustworthy, no generalization can be relied on.
3. **Adopt a pluggable optimization backend** (CVXPY / Riskfolio-Lib) behind the weighting interface to add a general convex-constraint layer (sector/group bounds, cardinality, tracking-error, turnover budget) and selectable objectives (MV / Black-Litterman / worst-case). The existing `BaseWeighting`/registry pattern is the right seam for this.
4. **Pluggable cost models** (linear → market-impact / fee-schedule / financing) behind a `CostModel` interface — and apply them on *both* the single- and multi-period paths (today single-period ignores transaction costs, §3.6).
5. **Pluggable regime models** — make the current binary threshold one implementation of a `RegimeModel` interface that can also host HMM/GMM/multi-state classifiers.
6. **Point-in-time / survivorship-safe universe** — represent manager universe membership as-of each rebalance date so backtests can't select managers that didn't yet exist or only survived.
7. **Statistical-rigor layer** — deflated Sharpe, multiple-testing correction, and out-of-sample significance tests as first-class outputs (see 7.2).
8. **Daily-cadence + order-aware path** (longer-term) if the tool moves toward tradeable instruments rather than monthly manager returns.

### 7.2 Questions to ask before presenting this to colleagues

These are the diligence questions a quantitatively-sophisticated colleague will ask first. Several are **framed as verification tasks**, not asserted defects — they should be checked against the code/methodology before publishing:

1. **Selection bias / data-snooping (the #1 credibility question).** Ranking top-N managers by Sharpe across a large universe and then reporting that Sharpe *mechanically overstates skill*. Do we correct for multiple testing (e.g. **deflated Sharpe ratio**, Bonferroni/BH), or hold out a true validation set? Without this, an out-of-sample outperformance claim is fragile. *(Note: §3.5 found ranking uses rf=0 by default, which further distorts the Sharpe-based selection.)*
2. **Survivorship & backfill bias.** Is the manager universe **point-in-time**, or does it include only survivors / backfilled track records? Trend/CTA databases are notorious for both; this can manufacture most of a backtest's edge.
3. **Look-ahead in selection/weighting.** Is each rebalance's selection and weighting computed **strictly from data available at that date**? The walk-forward engine exists — but confirm the *score* and the *regime threshold* aren't fit on the full sample (the §3.4 regime findings make this worth verifying explicitly).
4. **Gross vs. net of fees.** Are the input manager returns gross or net? Are management/performance fees modeled? The flat `monthly_cost` is too crude to defend a net-return claim.
5. **Statistical significance of the result.** The bootstrap CIs exist — are the selected portfolio's out-of-sample gains over the honest baselines (1/N of all managers, equal-weight top-N, SG Trend Index) **outside the confidence bands**, or within noise?
6. **What is the honest benchmark?** Equal-weight-all vs. equal-weight-top-N vs. a CTA peer index — the choice of baseline can flip the conclusion.
7. **Parameter stability.** How sensitive are results to `top_n`, `score_by`, `lookback`, and rebalance cadence? A result that only works at one parameter set is overfit.
8. **Investability / capacity.** Can you actually allocate to the selected managers (minimums, closed funds, liquidity/lockups, subscription-redemption mechanics)? Fund "transaction costs" differ fundamentally from the trading-cost bps model.
9. **Regime overlay validity.** Given §3.4 (`threshold=0.0` degenerate in vol mode; `annualise_volatility` silently shifts the boundary), is the regime split economically meaningful and out-of-sample, or an artifact of defaults?
10. **Frequency honesty.** Given §3.2, confirm what cadence the numbers actually reflect before quoting any annualized figure — today daily/weekly inputs are silently monthly with √12 annualisation.

---

## Appendix — methodology & coverage notes

**How this audit was run.** A capacity-aware mix of background multi-agent workflows and direct inspection:

- **Economics/wiring (§3)** — a fan-out workflow inventoried every config parameter (definition + consumption sites) per domain, judged wiring + economic sensibility, then adversarially verified flagged findings. The `portfolio.*` domain failed twice as a workflow (once an agent stall, once a forced-schema miss on `sonnet`), so it was **audited directly** by the lead auditor with targeted reads/greps — slightly less fan-out, but the key claims (weighting bifurcation, cost path, single-period cost gap) were each confirmed against source.
- **Landscape (§5)** — a research workflow with primary-source-verified summaries across 28 tools in 5 categories, then a synthesis pass. Point-in-time facts flagged in §5.5.
- **Quality/dedup/UX (§1, §2, §4)** — a workflow with three parallel finders + an adversarial verify pass (one of six verifiers stalled; 5 landed).
- **Strategy synthesis (§6, §7)** — written directly by the lead auditor, grounded in §3 + §5.

**Independence.** Per the agreed scope, the pre-existing `review-suggested-issues.md` was **not** read or used as input by any agent — this is a fully independent pass.

**Verification discipline.** Findings flagged as wiring bugs / not-wired / economically-questionable / P0-P1 code issues were handed to an independent agent instructed to *refute* them; only confirmed (or severity-corrected) findings are reported. Notable corrections made by verification: several P1s were down-graded to P2/info, the `config/model.py` vs `config/models.py` "duplication" was cleared as justified separation, and the weighting trap was scoped to YAML/CLI (the GUI was verified safe).

**Coverage caveats (what was *not* done):**

1. **UX review is static** — the live Streamlit app was **not** launched. A live click-through is recommended to confirm the §4 findings (esp. the dead Validation page and the Monte-Carlo IA) and to assess responsiveness/visual polish that source can't show.
2. **Not a line-by-line review.** The repo has ~1,000 `.py` files; this is a *prioritized* audit focused on the core app per the agreed scope (`src/trend_analysis/`, `streamlit_app/`, CLI, `config/`, + supporting `scripts/`/`tools/`). Excluded by agreement: `agents/`, `archives/`, `retired/`, and `.github` automation (owned upstream by `stranske/Workflows`).
3. **Verify caps (no silent truncation):** economics flagged 30 findings → top 9 verified; quality/dedup → top 6 attempted (5 landed). Unverified findings carry finder-confidence only and are marked as such.
4. **Medium-confidence items to double-check before acting:** the "`ScorePropSimple` is unreachable from config" claim (only the two engine resolvers were traced); the exact behavior of the `sample_split.in_end` boundary convention in edge cases; and all point-in-time competitive facts in §5.

**Durable evidence artifacts** (raw structured findings, survive independent of this report):
- [reports/audit/economics_raw.json](reports/audit/economics_raw.json) — full parameter inventory + findings + verdicts (§3 domains 0/2)
- [reports/audit/landscape_raw.json](reports/audit/landscape_raw.json) — 28-tool research + synthesis (§5)
- [reports/audit/quality_ux_raw.json](reports/audit/quality_ux_raw.json) — quality/dedup/UX findings + verdicts (§1/§2/§4)

**Suggested next step:** convert the actionable findings into tracked issues (the config-honesty fix, the `data.frequency`/`data.date_column` fixes, the statistical-rigor layer, and the `backtesting/harness.py` consolidation are the highest-leverage). I did not file issues — per the agreed deliverable this is a single consolidated report.
