# Trend Model Project — Phase 2 Autogen Spec

Welcome, code‑gen agent!  
The scaffolding you see in the repo is intentional.  
**Fill only the TODO markers below.**

---

## Repository map (relevant pieces)

trend_analysis/
pipeline.py # contains single_period_run(data, cfg, ...)
multi_period/
engine.py # ← implement run(cfg)
scheduler.py # complete
replacer.py # ← implement Rebalancer
gui/app.py # streamlit placeholder
config/defaults.yml # config schema

markdown
Copy

---

## Tasks for this agent

### T1 `trend_analysis/multi_period/replacer.py`
Implement class **`Rebalancer`**

* `__init__(self, cfg)`
* `apply_triggers(prev_weights, score_frame) -> pd.Series`
  * Enforce min/max funds from `cfg.multi_period`.
  * Apply σ‑trigger logic:
      * z‑score < −σ for N consecutive periods → mark for removal
      * z‑score > +σ this period & portfolio below max_funds → consider addition
  * Preserve prev weights unless traded; renormalise to 1.00.

### T2 `trend_analysis/multi_period/engine.py`
Implement `run(cfg) -> Dict[str, SingleRunResult]`:

1. Generate schedule via `generate_periods(cfg)`.
2. Loop chronologically:
   * For first period, start equal‑weighted.
   * Call `single_period_run`.
   * Pass its `score_frame` + `prev_weights` to `Rebalancer.apply_triggers`.
   * Save checkpoint Parquet in `cfg.checkpoint_dir`.
3. Accumulate per‑period results + summary frame.
4. Return `{"periods": results_dict, "summary": summary_df}`.

### T3 Extend `tests/`
* Unit tests for `Rebalancer.apply_triggers`.
* Integration test: 3‑period × 5‑fund dummy → `engine.run(cfg)` returns summary.

---

## Constraints
* Code must pass `ruff` & `black`.
* Only use numpy, pandas, joblib, streamlit.
* Keep functions pure; use `cfg.random_seed` for reproducibility.

Happy coding! 🚀
