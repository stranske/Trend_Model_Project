# App behavior baseline kit (TMP pilot)

A systematic way to test the app's inputs, confirm the output is economically
sensible, and store a **blessed baseline** that future versions are compared
against. It answers three questions that ordinary unit tests missed here:

| Question | Tier | How |
|---|---|---|
| Is each input control actually *wired* to the logic? | 2 | `test_wiring.py` — flip a flag, require output to change |
| Is the output *economically sensible*? | 1, 3 | `test_directional.py` (changes move the right way) + `test_invariants.py` (always-true rules) |
| Does a new version still match the *blessed* output? | 0, 1 | `test_golden.py` — stored baselines diffed with tolerance |

## What you edit

**`catalog.yaml`** is the human source of truth — *what* we test. Add a
scenario or toggle there; no Python changes needed. Each Tier-1 scenario runs a
`control` and a `vary` config and checks a metric moves in the expected
`direction`:

- `enforce: true`  → a wrong direction **fails** (high-confidence economics).
- `enforce: false` → the observed direction is **reported, not asserted** —
  used while we're still confirming what "sensible" means, or when the current
  demo config doesn't exercise that knob.

**`invariants.py`** holds the always-true economic rules (weights sum/cap,
long-only, vol ≥ 0, Sharpe soft-band ≤ 5, etc.). Edit the bounds there.

## Running

```bash
# Full suite (deterministic; pin the hash seed)
PYTHONHASHSEED=0 pytest tests/baseline/ -n0

# Re-bless golden baselines after an INTENTIONAL change, then review the diff:
PYTHONHASHSEED=0 pytest tests/baseline/test_golden.py -n0 --force-regen
git diff tests/baseline/test_golden/      # inspect, then commit
```

Baselines live in `tests/baseline/test_golden/*.csv` and are committed to git.
A failing golden test means output changed — either a regression (investigate)
or an intended change (re-bless and commit).

## Outputs

- `docs/reports/baseline-coverage.md` — the **input-coverage manifest**: which
  schema parameters are exercised by a scenario, which the engine actually read
  at runtime, and any priority gaps. Written under `docs/reports/` and committed
  so the weekly repo-review evaluator discovers it and can raise
  "untested/unwired parameter" issues.

## Layout

```
harness.py                # load config -> apply patch -> run -> normalized output
catalog.yaml              # scenario + toggle definitions (edit this)
invariants.py             # Tier-3 economic rules
manifest.py               # schema -> scenario coverage (the custom piece)
test_golden.py            # Tier 0/1 golden masters (pytest-regressions)
test_directional.py       # Tier 1 directional/metamorphic checks
test_wiring.py            # Tier 2 wiring checks
test_invariants.py        # Tier 3 invariants on baseline + every scenario
test_coverage_manifest.py # emits coverage.md; guards catalog quality
```

## Findings (first run 2026-05-30; verdicts after code trace)

| # | Finding | Verdict |
|---|---|---|
| 1 | Holdings fixed at 20 regardless of `rank.n` / `selector.top_n` / `selection_mode` | **Open — intent clarified, fix pending.** Owner policy: in *fixed-count* mode `rank.n` is irrelevant (the fixed count governs); in *variable-count* mode the ranked selection count must MATCH that period's target holdings. Current multi-period path applies neither (holds all 20). Root cause not yet definitively pinned (the multi-period selection-count linkage); needs a careful, results-affecting fix to core selection logic. `rank_n_down` stays report-only until fixed. |
| 2 | `regime.enabled` toggle is a no-op in the demo fixture | **Fixture gap** — scalar `portfolio.max_turnover` only bypasses regime-specific cap lookup; it does not disable regime-aware selection or weight overrides. The demo fixture does not yet exercise a regime-sensitive allocation path, so this toggle remains report-only until a regime-conditional scenario is added. |
| 3 | `rf_rate_annual` doesn't move Sharpe | **FIXED 2026-05-30** — per-fund metrics hardcoded rf=0 (export/__init__.py); now use aggregated rf_in/rf_out. `rf_up` scenario enforced. |
| 4 | `max_weight=0.03` → max weight 0.0476 | **Open — needs scope decision.** When `max_weight × N < 1` the model silently scales weights *past* the cap instead of holding cash. A hard "infeasible → error" guard was tried but reverted: it breaks ~47 tests and blocks legitimately small portfolios (e.g. 2 funds at 25% cap, which should just hold ~50% cash). Choice: (a) error only when full investment is explicitly required, (b) respect cap + hold cash, (c) error on all cap×N<1. Invariant stays feasibility-aware. |
| 5 | Fund weights sum to ~0.95 | **Not a bug** — intentional cash allocation, tracked as `cash_weight` (portfolio.py:558). |
| 6 | Model page crashes on cold render | **FIXED 2026-05-30** (caught by AppTest) — nested-expander crash inside the "Config Chat" expander. Demoted the timing/NL-log expanders in `2_Model.py` AND the Replay expanders in `components/nl_operation_viewer.py` to `st.container`. (The state-dependent second pair was missed on the first pass; the AppTest guard now covers it.) Passing regression guard. |

### Harness refinements queued (from findings)
- rf check should read the *reported* Sharpe, not the rf=0 derived one (Finding 3).
- weight-sum invariant should be cash-aware: `weight_sum + cash_weight ≈ 1` (Finding 5).
- add a regime-sensitive scenario so multi-period regime allocation and
  turnover-cap wiring are exercised (Finding 2).

## Not yet done (planned)

- Extracting the generic harness into a shared, pip-installable pytest plugin
  for reuse across apps (trip-planner next).
- Wiring `reports/baseline/coverage.md` into the weekly repo-review issue
  automation (a Workflows-repo change).

Done: Streamlit `AppTest` page smoke (`test_streamlit_smoke.py`).
```
