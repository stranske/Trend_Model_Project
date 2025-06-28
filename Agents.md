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
## Task #1 – Implement `multi_period/replacer.py::Rebalancer`

### Context
* `prev_weights`: `pd.Series` indexed by `fund_id`, weights from **previous** period (summing to 1.0).  
* `score_frame`: `pd.DataFrame` returned by `single_period_run`; must contain columns:  
  * `zscore` (performance z‑score vs. peer mean)  
  * `rank`   (1 = best)  
* Config path: `cfg["multi_period"]`.

### Requirements
1. **Constructor**
   ```python
   def __init__(self, cfg):
       self.cfg = cfg
       mp = cfg["multi_period"]
       self.min_funds = mp["min_funds"]
       self.max_funds = mp["max_funds"]
       self.triggers = mp["triggers"]      # dict as per defaults.yml
       self.anchors  = mp["weight_curve"]["anchors"]

2.apply_triggers(prev_weights, score_frame) → pd.Series
Identify removals
For each fund in prev_weights whose zscore < −σ, accumulate a strike.
Maintain an in‑memory self._strike_table (fund → consecutive_low_count).
Remove the fund if it hits its configured strike threshold (σ & periods).
Identify additions
Funds not in the portfolio whose zscore > +σ and portfolio size < max → eligible candidates.
Sort candidates by rank and add until max_funds reached.
Reweighting logic
Start with prev_weights for surviving funds.
For each holding, compute a multiplier via linear interpolation of anchors on rank percentile.
Multiply weights by those multipliers.
Assign equal weight to any new entrants.
Renormalise the final series to sum to 1.0.

Edge cases
If removals push holdings below min_funds, stop removing and issue a warnings.warn().
If prev_weights is empty (first period), return equal weights for the top‑ranked funds up to min_funds.

Random seed
Use np.random.default_rng(cfg["random_seed"]) for tie‑break orders when ranks equal.

Docstrings & typing
Follow existing style (numpy‑style docstrings).
Unit tests (to live in tests/test_replacer.py)
Test removal after consecutive lows.
Test addition when high z‑score and room in portfolio.
Test weight normalisation within 1e‑9 tolerance.

Constraints
Only Pandas, NumPy, and Python standard lib.
Must pass ruff and black --check.
No global state except the strike table inside the Rebalancer instance.

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
