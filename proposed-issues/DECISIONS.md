# Final recommendations on the conflicts & competing approaches

_After the follow-on review (wiring + sync-ownership verified against the current `phase-3` checkout)._

## Decision 1 — `run_analysis.py` / `run_multi_analysis.py`: **delete (T10), drop refactor (A23)**

**Verified facts:**
- `[project.scripts]` maps `trend-analysis`/`trend-multi-analysis` → `trend.compat_entrypoints:trend_analysis/trend_multi_analysis` ([pyproject.toml:56-57](pyproject.toml:56)).
- `trend/compat_entrypoints.py` does `from trend import cli as trend_cli` and delegates there — it does **not** import or call `trend_analysis.run_analysis` / `run_multi_analysis` ([compat_entrypoints.py:6](src/trend/compat_entrypoints.py:6)).
- No `src/` production module imports either runner (the many `_run_analysis` hits are the unrelated pipeline function). `run_multi_analysis` is a public lazy submodule ([__init__.py:168](src/trend_analysis/__init__.py:168)) with no production caller; `run_analysis.py` isn't even in the lazy loader.
- Dependents are ~15 test files + `scripts/run_multi_demo.py`.

**Recommendation: adopt T10 (remove), drop A23 (refactor).** Deduping dead code is wasted effort — my A23 assumed these were live entry points; the wiring check shows they are not. **Caveat:** this is not a one-line delete — it requires migrating the ~15 dependent tests and `run_multi_demo.py` to the public API (`trend_analysis.api` / `trend.cli`) and removing the `_LAZY_SUBMODULES`/`__all__` entry. File as **C1** (below). *(Matches compendium findings #22/#23.)* **Confidence: high** on wiring, **medium** on migration effort.

## Decision 2 — T9 CLI-helper dedup: **scope to the 3 verified copies; verify-before-touch the rest**

**Verified facts:** `_apply_trend_spec_preset`, `_apply_universe_mask`, `_attach_universe_paths` are defined in **both** `src/trend/cli.py` and `src/trend_analysis/cli.py`. `_resolve_returns_path`, `_run_pipeline` (and per the compendium, `_ensure_dataframe`, `_print_summary`, `_write_report_files`) are also present in both but classified as **delegators**. The compendium is internally inconsistent (cli-duplicate-cleanup.md claims "8 duplicates"; resolve-returns-path.md/cli.md call several delegators); `review-suggested-issues.md` Appendix A resolved it: only the 3 are byte-equivalent.

**Recommendation: scope T9 to the 3 named helpers.** Make the issue's **first task** a verification step — diff the function bodies to confirm byte-equivalence of the 3 and confirm the other 5 delegate — then consolidate only confirmed copies. **Lesson adopted:** verify copy-vs-delegator before deduping (applies equally to my A7/A22). **Confidence: high.**

## Decision 3 — A1 + T3 (config validation): **complementary — do both**

`A1` makes inert/unknown keys fail loudly (`extra="forbid"` / declared-vs-consumed lint); `T3` makes `config/bridge.py::validate_payload` actually run Tier-2 `validate_trend_config` so `vol_adjust`/`max_turnover`/`selection_mode` are checked ([bridge.py:68](src/trend_analysis/config/bridge.py:68)). Different halves of the same gap. **Keep as two issues; sequence behind tests** (tightening may reject configs silently accepted today). **Confidence: high.**

## Decision 4 — silent-coercion cluster (A3, A6, T5): **three issues, one banner**

Three instances of the same anti-pattern: `data.frequency`→monthly (A3), `weighting.name`→equal (A6), `missing_policy` bfill→ffill (T5, [util/missing.py:75](src/trend_analysis/util/missing.py:75)). **Keep as separate issues** (distinct sites) **under a shared label** `no-silent-coercion` with a one-line policy: *honor the configured value or reject it — never silently substitute.* **Confidence: high.**

## Decision 5 — ownership (T8, T19, T14): **route synced files upstream**

**Verified against `Workflows/.github/sync-manifest.yml`:** `scripts/langchain/*` **is synced** from Workflows (incl. `pr_verifier.py` line 546, `injection_guard.py` line 519). `src/trend_analysis/{proxy,api_server,llm_proxy,llm}/` are **not** in the manifest → **repo-owned.**

- **T8** (`pr_verifier.py` missing `api_client`): **file UPSTREAM in `stranske/Workflows`** — a local fix is overwritten on next sync. Consumer-side action: add an import smoke test for `scripts/langchain/*` and re-sync once the upstream fix lands.
- **T19 → SPLIT:** **T19a (repo, here):** network-bind defaults in `api_server/__main__.py`, `proxy/cli.py`, `proxy/server.py`, `llm_proxy/server.py`. **T19b (upstream):** the `injection_guard.py` `re.DOTALL` regex fix.
- **T14** (`llm/chain.py` dedup): `llm/` is **repo-owned** → fix here. **Confidence: high.**

## New issue from the conflict

- **C1** — Remove orphaned legacy runners `run_analysis.py` / `run_multi_analysis.py`; migrate the ~15 dependent tests + `scripts/run_multi_demo.py` to the public API; drop the `_LAZY_SUBMODULES`/`__all__` entry. (Supersedes A23.)

**Net effect on the issue set:** drop **A23**; add **C1**; split **T19** → **T19a/T19b**; mark **T8** upstream; scope **T9** to 3 + verify-first.
