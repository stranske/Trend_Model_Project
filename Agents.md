# agents.md
"""
YOU ARE CODEX.  EXTEND THE VOL_ADJ_TREND_ANALYSIS PROJECT AS FOLLOWS
--------------------------------------------------------------------

## Demo pipeline (maintenance / CI)

1. **Bootstrap environment**

   ```bash
   ./scripts/setup_env.sh
   ```

2. **Generate demo dataset**

   ```bash
   python scripts/generate_demo.py
   ```

3. **Run full demo pipeline and export checks**

   ```bash
   python scripts/run_multi_demo.py
   ```

   The script must invoke `export.export_data()` with the demo results so CSV,
   Excel, JSON **and TXT** outputs are generated in one call.  Update it
   whenever new exporter functionality is added.

   When exporter features evolve (e.g. additional formats or option flags),
   extend both `run_multi_demo.py` and `config/demo.yml` so the demo pipeline
   exercises every new code path. This keeps CI in lock‑step with the live
   exporter behaviour.

4. **Run the test suite**
   ```bash
   ./scripts/run_tests.sh
   ```

5. **Keep demo config current**
   - Update `config/demo.yml` and demo scripts whenever export or pipeline
     behaviour changes so that the demo exercises all features.

See **[docs/DemoMaintenance.md](docs/DemoMaintenance.md)** for a concise
checklist of these steps.

High‑level goal
~~~~~~~~~~~~~~~
Add a **performance‑based manager‑selection mode** that works alongside the
existing 'all', 'random', and 'manual' modes.  Make the pipeline fully
config‑driven and keep everything vectorised.

Functional spec
~~~~~~~~~~~~~~~
1.  New selection mode keyword:  `rank`.
    • Works on the *in‑sample* window **after** the usual data‑quality filters.  
    • Supported inclusion approaches:
         - `'top_n'`       – keep the N best funds.
         - `'top_pct'`     – keep the top P percent.  
         - `'threshold'`   – keep funds whose score ≥ user threshold
           (this is the “useful extra” beyond N and percentile).

2.  Rank criteria (`score_by`):
    • Any single metric registered in `METRIC_REGISTRY`
      (e.g. 'Sharpe', 'AnnualReturn', 'MaxDrawdown', …).  
    • Special value `'blended'` that combines up to three metrics with
      user‑supplied *positive* weights (weights will be normalised to 1.0).

3.  Direction‑of‑merit:
    • Metrics where “larger is better”  → rank descending.
    • Metrics where “smaller is better” (currently **only** MaxDrawdown)  
      → rank ascending.  Future metrics can extend `ASCENDING_METRICS`.

4.  Config file (YAML) drives everything – sample below.

5.  UI flow (ipywidgets, no external deps):
    Step 1  – Mode (‘all’, ‘random’, ‘manual’, **‘rank’**),
               checkboxes for “vol‑adj” and “use ranking”.  
    Step 2  – If mode == 'rank' **or** user ticked “use ranking”
               → reveal controls for `inclusion_approach`,
               `score_by`, `N / Pct / Threshold`, and (if blended)
               three sliders for weights + metric pickers.  
    Step 3  – If mode == 'manual'  
               → display an interactive DataFrame of the IS scores so the
               user can override selection and set weights.
    Step 4  – Output format picker (csv / xlsx / json) then fire
               `run_analysis()` and `export_to_*`.

6.  No broken changes:
    • Default behaviour (config absent) must be identical to current build.
    • All heavy computation stays in NumPy / pandas vector land.

7.  Unit‑test hooks:
    • New pure functions must be import‑safe and testable without widgets.
      (e.g. `rank_select_funds()`).

Sample YAML
~~~~~~~~~~~
selection:
  mode: rank               # all | random | manual | rank
  random_n: 12             # only if mode == random
  use_vol_adjust: true
rank:
  inclusion_approach: top_n     # top_n | top_pct | threshold
  n: 8                          # for top_n
  pct: 0.10                     # for top_pct  (decimal, not %)
  threshold: 1.5                # ignored unless approach == threshold
  score_by: blended             # Sharpe | AnnualReturn | … | blended
  blended_weights:
    Sharpe: 0.5
    AnnualReturn: 0.3
    MaxDrawdown: 0.2
output:
  format: excel                 # csv | excel | json
"""

"""
🔄 2025‑06‑15 UPDATE — PHASE‑1 ENHANCEMENTS
------------------------------------------
• Blended ranking **must** use *z‑scores* (mean‑0, stdev‑1) before the
  weighted sum so metrics on different scales are commensurable.
• MaxDrawdown is currently the only “smaller‑is‑better” metric; the
  ASCENDING_METRICS set remains {"MaxDrawdown"} until further notice.
• Config format stays YAML.
"""# agents.md
## Mission
Converge the scattered modules into one fully‑test‑covered, vectorised pipeline that can be invoked from a single CLI entry‑point.
Never touch notebooks living under any directory whose name ends in old/.

---

### 2025-06-27 UPDATE — RISK-METRICS EXPORT (SERIOUSLY, LEAVE THIS IN)

Codex removed the pretty reporting layer once; it shall not happen again.  
Follow these guard-rails whenever you touch export logic.

1. **Call the canonical exporters**  
   After `pipeline.run()` completes, pipe the returned `Mapping[str, pd.DataFrame]`
   into **exactly one** of  
   `trend_analysis.export.export_to_excel | export_to_csv | export_to_json`.

2. **Sheet / file keys**  

3. **Excel format contract**  
* Generate the summary sheet formatter via
  `trend_analysis.export.make_summary_formatter(...)`.
* Register any other sheet formatter with `@register_formatter_excel`
  so `export_to_excel` auto-hooks it.
* Required cosmetics:  
  - bold title row,  
  - `0.00%` for CAGR & Vol,  
  - `0.00` for Sharpe & Sortino,  
  - red numerals for MaxDD,  
  - freeze panes on header,  
  - auto-filter,  
  - column width = `max(len(header)) + 2`.

4. **Column order = law**  
Tests must fail if this order mutates.

5. **Config switches**  
`output.format` = `excel | csv | json`  
`output.path`   = prefix used by exporter (Excel auto-appends `.xlsx`).

6. **Tests**  
* In-memory smoke test: write to `BytesIO`, assert two sheets and cell
  `A1 == "Vol-Adj Trend Analysis"`.
* Regression test: `assert list(df.columns) == EXPECTED_COLUMNS`.

7. **Back-compat**  
Silent config = drop the fully formatted Excel workbook into `outputs/`
exactly as v1.0 did. Breaking that throws `ExportError`.

> 🛡️  If you rip out these formatters again, CI will chaperone you with a failing gate and a stern commit message.


## | Layer / concern                      | **Canonical location**                                                     | Everything else is **deprecated**                         |
| ------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Data ingest & cleaning**           | `trend_analysis/data.py` <br> (alias exported as `trend_analysis.data`)    | `data_utils.py`, helper code in notebooks or `scripts/`   |
| **Portfolio logic & metrics**        | `trend_analysis/metrics.py` (vectorised)                                   | loops inside `run_analysis.py`, ad‑hoc calcs in notebooks |
| **Export / I/O**                     | `trend_analysis/export.py`                                                 | the root‑level `exports.py`, snippets inside notebooks    |
| **Domain kernels (fast primitives)** | `trend_analysis/core/` package                                             | stand‑alone modules under the top‑level `core/` directory |
| **Pipeline orchestration**           | `trend_analysis/pipeline.py` (pure)                                        | any duplicated control flow elsewhere                     |
| **CLI entry‑point**                  | `run_analysis.py` **only** (thin wrapper around `trend_analysis.cli:main`) | bespoke `scripts/*.py` entry points                       |
| **Config**                           | `config/defaults.yml` loaded through `trend_analysis.config.load()`        | hard‑coded constants, magic numbers in notebooks          |
| **Tests**                            | `tests/` (pytest; 100 % branch‑aware coverage gate)                        |    —                                                      |
One concern → one module.
Replacements must delete or comment‑out whatever they obsolete in the same PR.

Immediate Refactor Tasks
Flatten duplications

Rename data_utils.py → trend_analysis/data.py, adjust imports, delete the original.

Migrate the contents of the top‑level exports.py into trend_analysis/export.py; keep only a re‑export stub for one minor release.

Turn the stray core/ directory into an importable sub‑package:
core/indicator.py → trend_analysis/core/indicator.py, etc.

Single pipeline

Implement trend_analysis/pipeline.py exposing a pure function
run(config: Config) -> pd.DataFrame.

run_analysis.py should parse CLI args, build a Config, pass it to pipeline.run, then handle pretty printing / file output only.

Config resolution

# trend_analysis/config.py
from pydantic import BaseModel
class Config(BaseModel):
    defaults: str = Path(__file__).with_name("..").joinpath("config/defaults.yml")
    # ...other validated fields...
def load(path: str | None = None) -> Config: ...

Env‑var override: TREND_CFG=/path/to/override.yml run_analysis ...

Dependency hygiene

Heavy imports (numpy, pandas, scipy) at top of each module are fine.

No circular imports. pipeline.py orchestrates; nothing imports it.

Tests

NOTE: Test fixtures must be text-serialised (CSV/JSON); no binary formats in PRs.

Require 100 % branch coverage on trend_analysis/* via pytest‑cov in CI.

Conventions & Guard‑rails
Vectorise first.
Falling back to for‑loops requires a comment justifying why vectorisation is impossible or harmful.

Public API (exported in __all__) uses US‑English snake‑case; private helpers are prefixed with _.

Notebook hygiene: any new exploratory notebook must start with the header
# 🔬 scratchpad – may be deleted at any time.

CI (GitHub Actions) stages to add:

lint  (ruff + black –‑check)

type‑check (mypy, strict)

test (pytest ‑‑cov trend_analysis ‑‑cov‑branch)

build‑wheel (tags only)

##NEW

### ✨ Task: Integrate `information_ratio` end‑to‑end  (#metrics‑IR)

**Motivation**  
Phase‑1 now includes a vectorised `information_ratio` metric.  
It is fully unit‑tested but not yet surfaced in the CLI / Excel export or
multi‑benchmark workflows.

---

#### 1.  Pipeline / Statistics

* [x] Extend `_Stats` dataclass with `information_ratio: float`.
* [x] In `_compute_stats()` compute `information_ratio(df[col], rf_series)`.
* [x] Ensure `out_stats_df` includes the new field.

#### 2.  Multi‑benchmark support

* [x] Accept `benchmarks:` mapping in YAML cfg, e.g.

```yaml
benchmarks:
  spx: SPX
  tsx: TSX

### 2025‑07‑03 UPDATE — STEP 4: surface a real `score_frame`

* **Add** `single_period_run()` to **`trend_analysis/pipeline.py`**  
  * **Signature**  
    ```python
    def single_period_run(
        df: pd.DataFrame,
        start: str,
        end: str,
        *,
        stats_cfg: "RiskStatsConfig" | None = None
    ) -> pd.DataFrame:
        ...
    ```
  * **Behaviour**
    1. Slice *df* to `[start, end]` (inclusive) and drop the Date column into the index.  
    2. For every metric listed in `stats_cfg.metrics_to_run` **call the public registry** (`core.rank_selection._compute_metric_series`) to obtain a vectorised series.  
    3. **Concatenate** those series column‑wise into **`score_frame`** (`index = fund code`, `columns = metric names`, dtype `float64`).  
    4. Attach metadata  
       ```python
       score_frame.attrs["insample_len"] = len(window)        # number of bars
       score_frame.attrs["period"] = (start, end)             # optional helper
       ```
    5. Return `score_frame` – *no side effects, no I/O*.

* **Update callers**
  * `pipeline._run_analysis()` should call `single_period_run()` once, stash the resulting frame in the returned dict under key `"score_frame"`, but **must not** change existing outputs or CLI flags.
  * Existing metrics‑export logic stays exactly as is.

* **Tests**
  1. **Golden‑master**: compare the new `score_frame` against a pre‑generated CSV fixture for a small sample set.  
  2. **Metadata**: `assert sf.attrs["insample_len"] == expected_len`.  
  3. **Column order** equals the order of `stats_cfg.metrics_to_run`; failing this should raise.

* **Performance / style guard‑rails**
  * Remain fully vectorised—no per‑fund Python loops.
  * Keep `single_period_run()` *pure* (no global writes, no prints).
  * Do **not** introduce extra dependencies; stick to `numpy` + `pandas`.

> Once the test suite passes with the new `score_frame`, proceed to steps 5‑7 (Selector & Weighting classes).


### Step 5 – Selector classes

| Class | Purpose | Key Inputs |
| ----- | ------- | ---------- |
| `RankSelector` | Pure rank‑based inclusion identical to Phase 1 behaviour, but exposed as a plug‑in. | `score_frame`, `top_n`, `rank_column` |
| `ZScoreSelector` | Filters candidates whose z‑score > `threshold`; supports negative screening by passing `direction=-1`. | `score_frame`, `threshold`, `direction` |

Both selectors must return **two** DataFrames:  
1. `selected` – rows kept for the rebalancing date  
2. `log` – diagnostic table (candidate, metric, reason) used by UI & tests.

---

### Step 6 – Weighting classes

| Class (inherits `BaseWeighting`) | Logic | YAML Config Stub |
| --- | --- | --- |
| `EqualWeight` | 1/N allocation across `selected`; rounds to nearest bps to avoid float dust. | `portfolio.weighting: {method: equal}` |
| `ScorePropSimple` | Weight ∝ positive score; rescales to 100 %. | `portfolio.weighting: {method: score_prop}` |
| `ScorePropBayesian` | Same, but shrinks extreme scores toward the cross‑sectional mean using a conjugate‑normal update. | `portfolio.weighting: {method: score_prop_bayes, shrink_tau: 0.25}` |

---

### Step 7 – Engine wiring

Minimal loop in `multi_period/engine.py`:

```python
for date in schedule:
    candidates  = selector.select(score_frames[date])
    weights     = weighting.weight(candidates)
    portfolio.rebalance(date, weights)
```

Step 8 – Config schema delta

```yaml
metrics:
  registry: [annual_return, downside_deviation, sharpe_ratio]

portfolio:
  selector:
    name: zscore            # or 'rank'
    params:
      threshold: 1.0        # σ
  weighting:
    name: score_prop_bayes
    params:
      shrink_tau: 0.25
```

Step 9 – Unit‑test skeletons

```
tests/
└─ test_selector_weighting.py
   ├─ fixtures/score_frame_2025‑06‑30.csv
   ├─ test_rank_selector()
   ├─ test_zscore_selector_edge()
   ├─ test_equal_weighting_sum_to_one()
   └─ test_bayesian_shrinkage_monotonic()
```

Golden‑master strategy identical to Phase‑1 metrics: pickle one known score_frame
and compare selector/weighting outputs bit‑for‑bit (tolerances < 1e‑9).

Step 10 – Docs housekeeping

Phase‑1 docs stay at docs/phase-1/Agents.md.
Phase‑2 docs live in docs/phase-2/Agents.md (this file).
Cross‑link at the top.

### 2025‑07‑04 UPDATE — MULTI-PERIOD METRICS EXPORT

Phase‑2 introduces a rolling back‑tester.  The intent is that metrics from each
period mirror the Phase‑1 Excel output.  The export helpers must therefore
collect the per‑period metric tables and emit one worksheet per period in the
workbook.  CSV and JSON exports should likewise produce one file per period
using the existing :func:`export.export_data` helpers.

### 2025‑07‑10 UPDATE — PER-PERIOD WORKBOOK DETAIL

The original design goal is that each back‑test period should produce an Excel
worksheet indistinguishable from the Phase‑1 summary.  CSV and JSON exports must
likewise emit one file per period.  This behaviour has yet to be fully realised
in code and tests.

### 2025‑07‑11 UPDATE — PER‑PERIOD SUMMARY TABLES

Implement helpers so every multi‑period run yields one workbook tab (or file)
per period containing the full Phase‑1 style summary table.  Excel sheets are
formatted via ``make_period_formatter`` while CSV/JSON outputs receive the same
rows using ``summary_frame_from_result``.

### 2025‑07‑12 UPDATE — COMBINED SUMMARY SHEET

Multi‑period exports must also include a ``summary`` sheet aggregating portfolio
performance across all periods.  The sheet uses the **same** format as the
per‑period tabs and is generated via ``make_period_formatter`` on a result
dictionary produced by ``combined_summary_result(results)``.  CSV/JSON formats
write a ``_summary`` file derived from ``summary_frame_from_result``.
