# Suggested issues from the iamkayleb repo review

> **Superseded historical planning snapshot.** Do not file or implement work from
> this document. The repository authority is
> [`docs/audits/collab-feedback-closeout.md`](docs/audits/collab-feedback-closeout.md),
> and the coordination-space corrective ledger is
> `Code/Audits/Trend_Model_Project/2026-08-22-feedback-completion-ledger.md`.

Derived from `trend-review-suggested-material-compendium.docx` (iamkayleb's subsystem
briefs, PRs #8–#53, Jan–May 2026), **after independent verification against the current
`phase-3` tree** (the review was authored against the `iamkayleb` fork). Each item below was
confirmed present in this repo with file:line evidence unless noted.

**How to use this file:** each `## Issue` is filing-ready (title = the heading text after the
number). When you're happy, they can be created on `stranske/Trend_Model_Project` with `gh`.
Read **Appendix A** (findings deliberately *not* filed) and **Appendix B** (Workflows
ownership) before working any P0-8 / P2-19 item — some belong upstream, not here.

Priorities: **P0** = correctness bug · **P1** = duplication / dead-code cleanup (the main ask)
· **P2** = hygiene.

---

## P0 — correctness bugs

### Issue 1 — Fix Sortino annualization in `backtesting/harness._compute_metrics`
- **Priority:** P0 · **Area:** metrics/backtesting · **Labels:** `bug`, `metrics` · **Source:** PR #42

**Problem.** In `_compute_metrics`, Sharpe is annualized correctly but Sortino is not. The
downside deviation is annualized on its own line, then divided into the *per-period* mean,
so the Sortino ratio ends up effectively divided by `√periods_per_year` — about 3.46× too
small for monthly data. The canonical implementation in
[`metrics/__init__.py::sortino_ratio`](src/trend_analysis/metrics/__init__.py:253) does it
correctly (`annual_return / annualized_downside_vol`).

**Evidence.**
- [src/trend_analysis/backtesting/harness.py:635](src/trend_analysis/backtesting/harness.py:635) `downside_std = downside.std(ddof=0) * np.sqrt(periods_per_year)`
- [src/trend_analysis/backtesting/harness.py:641](src/trend_analysis/backtesting/harness.py:641) `sortino = active_returns.mean() / downside_std ...`
- Sharpe one line above multiplies the *whole ratio* by `√ppy`; Sortino is missing that outer factor.
- `tests/backtesting/test_harness.py` only asserts the `"sortino"` key exists, never its value — which is why this went unnoticed.

**Proposed fix.** Mirror the Sharpe formula: `sortino = active_returns.mean() / downside.std(ddof=0) * np.sqrt(periods_per_year)` (per-period downside std × outer `√ppy`), or build an annualized numerator. Add a numeric assertion to `tests/backtesting/test_harness.py` (cross-check against `metrics.sortino_ratio` on the same series).

**Risk.** Low code change, but it changes a reported metric value — call it out in release notes / changelog.

---

### Issue 2 — `scripts/check_branch.sh` fails bash syntax (duplicate `then`)
- **Priority:** P0 · **Area:** dev tooling · **Labels:** `bug`, `ci`, `scripts` · **Source:** PR #49

**Problem.** A stray duplicate `then` makes the whole script a syntax error, so the full local
validation gate and the generated pre-push hook abort immediately.

**Evidence.** [scripts/check_branch.sh:222](scripts/check_branch.sh:222) ends `... $XDIST_FLAG" ""; then then`. `bash -n scripts/check_branch.sh` exits 2: `syntax error near unexpected token 'then'`. Invoked by `quality_gate.sh --full` and the `git_hooks.sh` pre-push hook.

**Proposed fix.** Delete the extra `then`. Add `bash -n` over `scripts/*.sh` to a lint step so this class of breakage is caught in CI.

**Risk.** Trivial. (Repo-local dev script, not synced — safe to fix here.)

---

### Issue 3 — `config/bridge.py::validate_payload` only runs Tier-1 validation
- **Priority:** P0 · **Area:** config · **Labels:** `bug`, `config` · **Source:** PR #29

**Problem.** `validate_payload` calls Tier-1 `validate_core_config`, which validates only
`data.*` and `portfolio.cost_model`. `vol_adjust`, `portfolio.rebalance_calendar`,
`portfolio.max_turnover`, and `portfolio.selection_mode` pass through unchecked — even though
`build_config_payload` in the same file injects several of them. The Streamlit app accepts and
forwards those fields silently. **Not fixed by the recent config commits (#5356/#5357).**

**Evidence.** [src/trend_analysis/config/bridge.py:68](src/trend_analysis/config/bridge.py:68) `validate_core_config(payload, ...)`; payload built at [bridge.py:46-57](src/trend_analysis/config/bridge.py:46) injects `rebalance_calendar`/`max_turnover`/`vol_adjust.target_vol`.

**Proposed fix.** Call Tier-2 `validate_trend_config` from `validate_payload` (it covers `vol_adjust` and the `portfolio` minimums), or compose Tier-1 + the missing Tier-2 checks. Add a test that an invalid `target_vol`/`max_turnover` is rejected at the bridge.

**Risk.** Medium — tightening validation may surface configs that were silently accepted before; stage behind tests.

---

### Issue 4 — `pages/8_Validation.py` never renders (three compounding defects)
- **Priority:** P0 · **Area:** streamlit · **Labels:** `bug`, `streamlit` · **Source:** PR #53

**Problem.** The developer validation page is dead:
1. Its only render call sits under `if __name__ == "__main__":` — Streamlit imports pages, so it never runs.
2. Even if fixed, it reads session key `"app_data"`, which is **never written** anywhere in `streamlit_app/`; the app stores returns under `"returns_df"`.
3. It calls the private `analysis_runner._execute_analysis`, bypassing the public cached `run_analysis`.
4. `st.set_page_config()` is called *inside* the render function (line ~529), after Streamlit calls have begun — will raise once #1 is fixed.

**Evidence.** [8_Validation.py:893](streamlit_app/pages/8_Validation.py:893), [:543](streamlit_app/pages/8_Validation.py:543), [:498](streamlit_app/pages/8_Validation.py:498); canonical key in [state.py:23](streamlit_app/state.py:23); public API [analysis_runner.py:596](streamlit_app/components/analysis_runner.py:596).

**Proposed fix.** Use the `_should_auto_render()` pattern other pages use; switch to `app_state.get_uploaded_data()` / `"returns_df"`; call public `run_analysis(...)`; hoist `set_page_config` to module top.

**Risk.** Low — page is currently non-functional, so there's nothing to regress.

---

### Issue 5 — `missing_policy: bfill`/`backfill` is silently coerced to `ffill`
- **Priority:** P0 · **Area:** data/pipeline · **Labels:** `bug`, `data` · **Source:** PR #42

**Problem.** `util/missing.py::_coerce_policy` maps `{"both","bfill","backfill"} → "ffill"`. A user
configuring backward-fill silently gets forward-fill, with no warning and no real bfill code path.
The same string raises `ValueError` in `io/market_data.py::_normalise_policy_value`, so the two
layers disagree on identical input.

**Evidence.** [src/trend_analysis/util/missing.py:75](src/trend_analysis/util/missing.py:75); strict counterpart [io/market_data.py:281](src/trend_analysis/io/market_data.py:281).

**Proposed fix.** Either implement real bfill (`series.bfill(limit=...)`) or raise on `bfill`/`backfill` to match the io layer. Drop the undocumented `both`/`zero_fill` aliases unless they're in the user-facing schema. Add a test.

**Risk.** Low–medium — anyone currently passing `bfill` and tolerating ffill will see behavior change; document it.

---

### Issue 6 — `run_real_model.py` crashes on RF/portfolio date mismatch (`.loc` → `.reindex`)
- **Priority:** P0 · **Area:** scripts · **Labels:** `bug`, `scripts` · **Source:** PR #50

**Problem.** `rf_series.loc[portfolio.index]` raises `KeyError` whenever the stitched OOS portfolio
includes a date outside the loaded risk-free range — a realistic backtest configuration.

**Evidence.** [scripts/run_real_model.py:121](scripts/run_real_model.py:121).

**Proposed fix.** `rf_aligned = rf_series.reindex(portfolio.index) if rf_series is not None else 0.0` (NaN-fills missing labels). Decide on NaN handling downstream (fill 0 or forward-fill).

**Risk.** Trivial.

---

### Issue 7 — `state.py::_values_equal` reports phantom `int`↔`float` diffs
- **Priority:** P0 · **Area:** streamlit · **Labels:** `bug`, `streamlit` · **Source:** PR #52

**Problem.** The `type(left) is not type(right): return False` short-circuit runs before the numeric
`math.isclose` branch, so `_values_equal(10, 10.0)` returns `False`. Model states round-trip through
JSON (`export_model_state`/`import_model_state`), where `10` may deserialize as `int` and `10.0` as
`float`; comparing a fresh state to an imported one reports spurious `"10 → 10.0 [type changed]"`.

**Evidence.** [streamlit_app/state.py:317](streamlit_app/state.py:317).

**Proposed fix.** Check the numeric (`numbers.Number`) `isclose` branch before the strict type comparison.

**Risk.** Low.

---

### Issue 8 — `scripts/langchain/pr_verifier.py` is unimportable (missing `api_client`)
- **Priority:** P0 · **Area:** agent intake · **Labels:** `bug`, `needs-ownership-check` · **Source:** PR #51
- **⚠ Check Workflows ownership first (Appendix B). Likely fixed upstream, not here.**

**Problem.** `from scripts import api_client` references a module that does not exist anywhere in the
repo, so `import scripts.langchain.pr_verifier` raises `ImportError` at load; the `create_issue`
call site is unreachable.

**Evidence.** [scripts/langchain/pr_verifier.py:26](scripts/langchain/pr_verifier.py:26) and `:681`; `find . -name 'api_client*'` → nothing.

**Proposed fix.** If `scripts/langchain/*` is synced from `stranske/Workflows`, `api_client.py` was probably never synced to this consumer — fix by adding it to the sync manifest / syncing it, **not** by writing a local module. Otherwise, restore/author `scripts/api_client.py`. Add an import smoke test for `scripts/langchain/*`.

**Risk.** Depends on routing — confirm ownership before touching.

---

## P1 — duplication & dead-code cleanup (the core focus)

### Issue 9 — Consolidate the 3 truly copy-pasted CLI helpers
- **Priority:** P1 · **Area:** cli · **Labels:** `refactor`, `duplication` · **Source:** PR #28 (reframed)

**Problem.** `trend/cli.py` and `trend_analysis/cli.py` contain three **byte-equivalent**
reimplementations: `_apply_trend_spec_preset`, `_apply_universe_mask`, `_attach_universe_paths`.
**Important:** the other five "duplicates" the review listed are already thin delegators, and the
`_load_configuration`/`_json_default` "copies" are delegators or genuinely different helpers — **do
not** dedupe those mechanically.

**Evidence.** Real copies: [trend/cli.py:134/163/193](src/trend/cli.py:134) vs [trend_analysis/cli.py:199/284/320](src/trend_analysis/cli.py:199). (Delegators to leave alone: `_resolve_returns_path`, `_ensure_dataframe`, `_run_pipeline`, `_print_summary`, `_write_report_files`.)

**Proposed fix.** Move the three into a shared module (e.g. a `trend_analysis` util or `trend.cli`) and have the other side import them, matching the existing delegator pattern.

**Risk.** Low if scoped to the 3 confirmed copies and covered by existing CLI tests.

---

### Issue 10 — Remove orphaned/unwired CLI modules and dead stub
- **Priority:** P1 · **Area:** cli/packaging · **Labels:** `cleanup`, `dead-code` · **Source:** PR #33, #38

**Problem.** Several modules are not wired and not used in production:
- `src/trend_analysis/run_analysis.py` — own argparse `main()`, not in `[project.scripts]`, not imported in `src/`, not in the lazy loader (only `tests/test_constants.py`).
- `src/trend_analysis/run_multi_analysis.py` — registered in `_LAZY_SUBMODULES` but no production caller.
- `src/cli.py` — `cv`/`report` subcommands, not in `[project.scripts]`, only test-imported; relies on `src/__init__.py` making `src` importable.
- `src/trend_portfolio_app/__init__.py` — empty compat stub (`__all__ = []`), no production callers.

**Proposed fix.** Delete `run_analysis.py` and `run_multi_analysis.py` (+ remove their `_LAZY_SUBMODULES`/`__all__` entries). Either wire `src/cli.py` as a real entry point or move it under `scripts/`; drop the `src/__init__.py` marker if no other `src.*` imports remain. Delete `trend_portfolio_app/` (update the handful of tests referencing it).

**Risk.** Low — confirm no `[project.scripts]`/import references before each deletion; some tests import these and will need updating.

---

### Issue 11 — Move CI-fixture files out of the production `trend_analysis` namespace
- **Priority:** P1 · **Area:** packaging/CI · **Labels:** `cleanup`, `ci` · **Source:** PR #33

**Problem.** Six intentional-violation CI fixtures live inside production source
`src/trend_analysis/`: `_autofix_probe.py`, `_autofix_trigger_sample.py`,
`_autofix_violation_case2.py`, `_autofix_violation_case3.py`, `_ci_probe_faults.py`,
`automation_multifailure.py`. They pollute the public namespace and IDE autocomplete.

**Proposed fix.** Move to `src/trend_analysis/_ci_fixtures/` (or `tests/fixtures/`) and update the ~5 `tests/workflows/` imports. No production code imports them, so low risk.

**Risk.** Low.

---

### Issue 12 — De-duplicate helpers in `export/__init__.py` and `viz/`
- **Priority:** P1 · **Area:** reporting/export/viz · **Labels:** `refactor`, `duplication` · **Source:** PR #43

**Problem / checklist (all verified):**
- `portfolio_series` defined **3× byte-identical** at [export/__init__.py:593/873/1366](src/trend_analysis/export/__init__.py:593) → extract one `_weighted_sum`.
- `_to_nav_wide` **2× byte-identical** in [viz/charts/rolling_panel.py:34](src/trend_analysis/viz/charts/rolling_panel.py:34) and `seasonality_heatmap.py` → import from `viz/adapters.py` (`_paths_to_wide_nav`).
- `_git_hash` **2×** with divergent `except` — [export/bundle.py:23](src/trend_analysis/export/bundle.py:23) (specific) vs [reporting/run_artifacts.py:28](src/trend_analysis/reporting/run_artifacts.py:28) (bare `Exception`) → hoist to `util/`, keep the specific catch.
- Dead `if proxy is not None: pass` at [export/__init__.py:1109](src/trend_analysis/export/__init__.py:1109) → delete.
- Two redundant openpyxl proxy/adapter class pairs (`_Openpyxl*Proxy` vs `_Openpyxl*Adapter`) → collapse to one.
- (Related notes) inconsistent bare `[]` vs `.get()` result-dict access in `_build_summary_formatter`/`format_summary_text` → use `.get(...)` like `summary_frame_from_result`.

**Proposed fix.** One PR with the checklist above; behavior-preserving.

**Risk.** Low–medium — `export/__init__.py` is 2,051 lines; lean on existing export tests and add a couple for the consolidated helpers.

---

### Issue 13 — De-duplicate `trend/mc` and reporting helpers
- **Priority:** P1 · **Area:** mc/reporting · **Labels:** `refactor`, `duplication` · **Source:** PR #39

**Problem.** `NAV_PATH_REQUIRED_CHARTS` is defined in both [mc/charts.py:6](src/trend/mc/charts.py:6)
(which even exports it) and [mc/viz.py:16](src/trend/mc/viz.py:16) — they can drift. `_init_matplotlib`
is near-duplicated (differs by 2 rcParams) across [reporting/quick_summary.py:31](src/trend/reporting/quick_summary.py:31)
and [reporting/unified.py:27](src/trend/reporting/unified.py:27).

**Proposed fix.** `viz.py` imports the constant from `charts.py`; extract a single `_init_matplotlib` into `reporting/_matplotlib.py` (reconcile the rcParam superset).

**Risk.** Low.

---

### Issue 14 — Refactor LLM chain duplication in `llm/chain.py`
- **Priority:** P1 · **Area:** llm · **Labels:** `refactor`, `duplication` · **Source:** PR #44
- **⚠ Confirm `src/trend_analysis/llm/*` is repo-owned (it appears to be — package code, not `tools/`).**

**Problem.** `ConfigPatchChain.run()` and `ConfigPatchVariantsChain.run()` are ~200 lines of
near-identical logic (prompt build → injection check → structured/text select → retry loop →
schema validate → `finally` logging); both already share `_BaseConfigPatchChain`. `ResultSummaryChain`
is a separate `@dataclass` that re-implements `_bind_llm`/`_invoke_llm`. `_read_env_float` is defined
identically in `chain.py` and `result_feedback.py`.

**Evidence.** [chain.py:443](src/trend_analysis/llm/chain.py:443), [chain.py:659](src/trend_analysis/llm/chain.py:659), `ResultSummaryChain` at `:852`; `_read_env_float` at [chain.py:997](src/trend_analysis/llm/chain.py:997) and [result_feedback.py:95](src/trend_analysis/llm/result_feedback.py:95).

**Proposed fix.** Extract a `_BaseConfigPatchChain._run_chain(prompt_builder, parser, ...)`; make the two `run()`s thin wrappers; let `ResultSummaryChain` reuse shared binding; hoist `_read_env_float` to a small `llm/_env.py`.

**Risk.** Medium — central LLM path; cover with the existing chain tests and add retry/injection regression cases.

---

### Issue 15 — Remove import-time side effects in `metrics/__init__.py`
- **Priority:** P1 · **Area:** metrics · **Labels:** `refactor`, `tech-debt` · **Source:** PR #41

**Problem.** Importing the metrics package monkey-patches `builtins` (`annualize_return`,
`annualize_volatility`) and registers a synthetic `tests.legacy_metrics` via
`sys.modules.setdefault` that collides with the real `tests/legacy_metrics.py` (resolution depends on
import order). The 439-line `__init__.py` mixes registry, metric impls, and these side effects.

**Evidence.** [metrics/__init__.py:431](src/trend_analysis/metrics/__init__.py:431) (builtins) and `:426` (`tests.legacy_metrics`).

**Proposed fix.** Drop the `builtins` patch (expose back-compat via an explicit module if still needed); pick one `tests.legacy_metrics` (keep the real file, drop the synthetic registration). Optionally split into `metrics/registry.py` + `metrics/core.py` + thin `__init__.py`.

**Risk.** Medium — verify the two `test_metric_vectorise*` tests still resolve `tests.legacy_metrics` correctly.

---

### Issue 16 — Hoist duplicated `_infer_periods_per_year` and reconcile drift
- **Priority:** P1 · **Area:** util · **Labels:** `refactor`, `duplication` · **Source:** PR #42

**Problem.** Defined in both [backtesting/harness.py:555](src/trend_analysis/backtesting/harness.py:555)
(ends `return max(1, approx)`) and [engine/walkforward.py:110](src/trend_analysis/engine/walkforward.py:110)
(has a `median_days <= 0` guard, lacks the `max(1, approx)` floor). The drift is a latent
behavioral divergence.

**Proposed fix.** Move to `util/frequency.py`, import from both, keep both guards (the `median_days <= 0` check *and* the `max(1, approx)` floor).

**Risk.** Low.

---

### Issue 17 — Stop importing deprecated shims from live Streamlit modules
- **Priority:** P1 · **Area:** streamlit · **Labels:** `cleanup`, `tech-debt` · **Source:** PR #52

**Problem.** `streamlit_app/config_bridge.py` and `components/date_correction.py` warn on import and
tell callers to use `trend_analysis.*`, but `guardrails.py`, `csv_validation.py`, and `pages/1_Data.py`
still import the shims — so the app emits its own `DeprecationWarning`s at startup. `analysis_runner.py`
already uses the canonical path, confirming it's a missed find-and-replace.

**Evidence.** [guardrails.py:22](streamlit_app/components/guardrails.py:22), [csv_validation.py:18](streamlit_app/components/csv_validation.py:18); canonical [analysis_runner.py:14](streamlit_app/components/analysis_runner.py:14).

**Proposed fix.** Repoint the three importers at `trend_analysis.config.bridge` / `trend_analysis.io.date_correction`, then delete the shims (or keep them only if external callers depend on them).

**Risk.** Low.

---

## P2 — hygiene (group as you see fit)

### Issue 18 — Deprecation & portability sweep
- **Priority:** P2 · **Area:** mixed · **Labels:** `cleanup`, `deprecation` · **Source:** PR #50, #53, #49

- `DataFrame.applymap` → `.map` at [3_Results.py:1370](streamlit_app/pages/3_Results.py:1370) (deprecated pandas 2.1).
- Dead `ast.Str`/`ast.Str.s` branch in `scripts/evaluate_settings_effectiveness.py` (~L141-146).
- `sed -i .bak` GNU/BSD portability in [scripts/test-release.sh:36](scripts/test-release.sh:36) (use `sed -i.bak` or a Python one-liner).
- Literal backticks from single-quoted `echo` in [scripts/open_pr_from_issue.sh:53](scripts/open_pr_from_issue.sh:53)/`:59` (PR body code fence renders broken).
- `$?`-after-pipeline in [scripts/quick_check.sh:23](scripts/quick_check.sh:23) (warning branch unreachable without `pipefail`).

**Note.** `datetime.utcnow()` items the review listed in `tools/coverage_guard.py` are **already fixed** in this tree — skip. The repo-local `scripts/*.sh` items above are safe to fix here.

**Risk.** Low; mostly one-liners.

---

### Issue 19 — Tighten network-bind and injection defaults
- **Priority:** P2 · **Area:** security · **Labels:** `security`, `hardening` · **Source:** PR #44, #47, #51
- **⚠ `injection_guard` and the `llm_proxy`/`tools` pieces may be Workflows-owned — see Appendix B.**

- Default host `0.0.0.0` → `127.0.0.1` for [api_server/__main__.py:7](src/trend_analysis/api_server/__main__.py:7), [proxy/cli.py:30](src/trend_analysis/proxy/cli.py:30), [proxy/server.py:237](src/trend_analysis/proxy/server.py:237), and `llm_proxy/server.py` (carries the upstream API key). Require an explicit `--host 0.0.0.0` for external reachability.
- `scripts/langchain/injection_guard.py` regex uses `re.IGNORECASE` but not `re.DOTALL`, so a newline between trigger words bypasses detection — add `re.DOTALL` (or `[\s\S]`). *(Likely Workflows-owned.)*

**Risk.** Low; behavioral default change — note in docs. Confirm ownership for the langchain piece.

---

### Issue 20 — Small UI correctness fixes
- **Priority:** P2 · **Area:** gui/ui · **Labels:** `bug`, `ui` · **Source:** PR #47

- Leading-space CSS property name in [gui/app.py:807](src/trend_analysis/gui/app.py:807) — `' --trend-theme'` should be `'--trend-theme'`; the theme toggle silently no-ops. Remove the leading space (and the pointless implicit string concat that hid it).
- [ui/rank_widgets.py:6](src/trend_analysis/ui/rank_widgets.py:6) imports `ipywidgets` unconditionally while `gui/app.py` treats it as optional — make the import lazy (inside `build_ui()`) so the module is importable in headless/CI environments.

**Risk.** Low.

---

## Appendix A — findings deliberately NOT filed

- **`llm/schema.py` "bare import" (PR #44/#45 #1) — WRONG.** `from utils.paths import proj_path` is
  the codebase-wide standard; `trend_analysis.utils.paths` does not exist, so the recommended fix would
  break the import. Do not action.
- **`coverage_guard.py` return-0-on-error and `datetime.utcnow()` (PR #48 #1/#2) — STALE.** Already fixed
  in this tree.
- **"8 duplicate CLI functions" / `_load_configuration` ×3 / `_json_default` ×4 (PR #28) — OVERSTATED.**
  Only the 3 helpers in Issue 9 are real copies; the rest are delegators or genuinely different. Acting
  literally would break working code.
- The review's many self-labeled "Notes (not actionable)" and "No issues (clean files)" entries are
  correctly low-priority and are not filed here; several were good *non*-actions (three-tier config,
  `CashPolicy` shim, `pipeline._run_analysis` patch target, `walk_forward` vs `walkforward` scopes).

## Appendix B — Workflows ownership (read before P0-8, P2-19, and any `tools/` work)

This is a **consumer repo**; per `CLAUDE.md`, `tools/*`, `.github/codex/*`, and synced scripts are
owned by `stranske/Workflows` and managed via `.github/sync-manifest.yml`. Verified-synced files among
the findings: `tools/coverage_guard.py`, `tools/langchain_client.py`, `tools/post_ci_summary.py`,
`tools/llm_provider.py`, `tools/enforce_gate_branch_protection.py`. `scripts/langchain/*` (incl.
`pr_verifier.py`, `injection_guard.py`, `structured_output.py`) is agent-intake infra and **may** be
synced too. **For anything synced, fix upstream in Workflows and re-sync — not here.** The repo-local
`scripts/*.sh` dev/release helpers and everything under `src/trend_analysis/`, `src/trend/`,
`streamlit_app/` are this repo's to fix.

## Appendix C — process feedback for the reviewer (not code)

- The compendium's Source Index shows ~12 "Exact duplicate of PR #X" resubmissions of the same briefs —
  ironic given the subject, and worth squashing in future deliverables.
- Early PRs (#8–#26) carried frequent typos and were descriptive walkthroughs rather than findings;
  quality climbs sharply from PR #33 on. The reviewer clearly grew into the work.
