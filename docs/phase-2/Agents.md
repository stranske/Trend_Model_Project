# agents.md
See [Phase 1 docs](../phase-1/Agents.md) for earlier context.
"""
YOU ARE CODEX.  EXTEND THE VOL_ADJ_TREND_ANALYSIS PROJECT AS FOLLOWS
--------------------------------------------------------------------

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
  > Notebook widgets now live in the optional module
  > `trend_analysis.ui.rank_widgets`. Import `build_ui()` from that module (or
  > call the thin wrapper `trend_analysis.core.rank_selection.build_ui()`)
  > inside notebook sessions where `ipywidgets` is installed. Core pipeline
  > modules remain import-safe even when those extras are missing.
<!-- STEP 0 START -->
### Step 0 – Config Loader & Editor  📝

| Purpose | Controls | Behaviour |
|---------|----------|-----------|
| **Load existing config** | `FileUpload(accept=".yml")` | Parse YAML → populate `ParamStore` → refresh downstream widgets. |
| **Template picker** | `Dropdown(options=list_builtin_cfgs())` | Selecting a template triggers the same refresh. |
| **Grid editor** | If **ipydatagrid** present: render editable grid of the current YAML.  Else show a disabled grid stub plus a warning banner. | Edits propagate to `ParamStore` in real time via the `on_cell_change` event; invalid edits revert and flash red. |
| **Save/Download** | “💾 Save config” button → writes YAML to disk; “⬇️ Download” → triggers browser download. | Uses `yaml.safe_dump(param_store.to_dict())`. |

> *Rationale*: power users often arrive with a ready config; making this the very first step short‑circuits half the clicks.

<!-- STEP 0 END -->
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
8.  Selector cache reuse:
    • Window bundles are keyed by `(start, end, universe_hash, stats_cfg_hash)` via
      `make_window_key()` and surfaced through `get_window_metric_bundle()`.
    • Instrument `selector_cache_stats()["selector_cache_hits"]` to verify that
      repeated selector runs on identical windows reuse the cached metrics and
      covariance payloads.

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
<!-- … existing Steps 1‑10 remain unchanged … -->

<!-- locate STEP 11 and replace its body with the following … -->

### Step 11 – GUI implementation (ipywidgets ± ipydatagrid)  🚀

> **Scope additions since v2**: Config‑first flow, true grid editing, `ParamStore`, debounce wrapper, state persistence, dark‑mode toggle, plug‑in registry.

| GUI Step | Mandatory Controls | Behaviour | Pure‑function hooks |
|----------|-------------------|-----------|---------------------|
| **0 – Config I/O** | See **Step 0** table above. | Valid edits update `ParamStore`; invalid edits rollback with toast. | `build_config_dict()` |
| **1 – Mode & global flags** | unchanged | unchanged | – |
| **2 – Ranking options** | unchanged | **New**: metric/weight sliders wrapped in 300 ms debounce decorator. | `rank_select_funds()` |
| **3 – Manual override** | **Primary**: `ipydatagrid.DataGrid` (editable include/weight columns).  <br>**Fallback**: previous SelectMultiple layout + warning. | Grid emits `cell_edited`; keeps weights numeric & ≥0. | – |
| **4 – Output & run** | + “🌗 Theme:” `ToggleButtons(["system","light","dark"])` | Dark‑mode switch toggles a CSS variable on the root DOM node. | – |
| **Status / logs** | unchanged | unchanged | – |

**ParamStore dataclass**

```python
@dataclass
class ParamStore:
    """Mutable GUI state shared across view layers."""
    cfg: dict[str, Any] = field(default_factory=dict)
    theme: str = "system"          # light | dark
    dirty: bool = False            # unsaved edits flag
    weight_state: dict[str, Any] | None = None  # AdaptiveBayes posterior

    def to_dict(self) -> dict[str, Any]:
        return self.cfg

    @classmethod
    def from_yaml(cls, path: Path) -> "ParamStore":
        return cls(cfg=yaml.safe_load(path.read_text()))

All widget callbacks accept (change, *, store: ParamStore) and mutate store
only; pipeline ingest path is run(build_config_from_store(store)).
Debounce helper
def debounce(wait_ms: int = 300):
    def decorator(fn):
        last_call = 0
        async def wrapper(*args, **kwargs):
            nonlocal last_call
            last_call = time.time()
            await asyncio.sleep(wait_ms / 1000)
            if time.time() - last_call >= wait_ms / 1000:
                return fn(*args, **kwargs)
        return wrapper
    return decorator

State persistence

On successful Run, call
yaml.safe_dump(store.to_dict(), Path.home()/".trend_gui_state.yml").

If ``store.weight_state`` is not ``None`` also pickle it to
``~/.trend_gui_state.pkl`` as ``{"adaptive_bayes_posteriors": store.weight_state}``.

On GUI launch, attempt to load the file; if malformed, ignore with a warning.

Plug‑in registry
for ep in importlib.metadata.entry_points(group="trend_analysis.gui_plugins"):
    plugin_cls = ep.load()
    register_plugin(plugin_cls)       # adds controls dynamically

Tests must assert that enumerating plug‑ins requires no widget edits.

<!-- STEP 11 END -->





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
| **Export / I/O**                     | `trend_analysis/export/` package                                           | the root‑level `exports.py`, snippets inside notebooks    |
| **Domain kernels (fast primitives)** | `trend_analysis/core/` package                                             | stand‑alone modules under the top‑level `core/` directory |
| **Pipeline orchestration**           | `trend_analysis/pipeline.py` (pure)                                        | any duplicated control flow elsewhere                     |
| **CLI entry‑point**                  | `trend/cli.py` via `trend`                                                 | bespoke `scripts/*.py` entry points                       |
| **Config**                           | `trend_analysis/config/` package with `models.py`                          | hard‑coded constants, magic numbers in notebooks          |
| **Tests**                            | `tests/` (pytest with coverage gate)                                       |    —                                                      |

One concern → one module.
Replacements must delete or comment‑out whatever they obsolete in the same PR.

### Refactor Status ✅

The following refactors from the original Phase-2 spec are **complete**:

- ✅ `data.py` – canonical data ingest module
- ✅ `export/` – package with formatters and multi-period helpers
- ✅ `core/` – importable sub-package with rank selection and metric cache
- ✅ `pipeline.py` – pure orchestration function
- ✅ `config/` – Pydantic models with YAML loading
- ✅ Env-var override: `TREND_CFG=/path/to/override.yml`

### Dependency Hygiene

Heavy imports (numpy, pandas, scipy) at top of each module are fine.

Keep formatter/test tool versions in lock-step by running `python -m scripts.sync_tool_versions --check` (use `--apply` if drift is detected) before committing config or workflow changes.

No circular imports. pipeline.py orchestrates; nothing imports it.

### Tests

NOTE: Test fixtures must be text-serialised (CSV/JSON); no binary formats in PRs.

Coverage gate enforced via pytest-cov in CI.

### Conventions & Guard‑rails
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
| `EqualWeight` | 1/N allocation across `selected`; rounds to nearest bps to avoid float dust. | `portfolio.weighting: {name: equal, params: {}}` |
| `ScorePropSimple` | Weight ∝ positive score; rescales to 100 %. | `portfolio.weighting: {name: score_prop, params: {}}` |
| `ScorePropBayesian` | Same, but shrinks extreme scores toward the cross‑sectional mean using a conjugate‑normal update. | `portfolio.weighting: {name: score_prop_bayes, params: {shrink_tau: 0.25}}` |

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

<!-- STEP 12 START -->
---

## Step 12 – Adaptive Bayesian Cross‑Period Weighting  🎯

> **Problem**  
> Current weighting schemes (`equal`, `score_prop_*`) “reset” every rebalance;
> persistent skill is ignored.  
> **Solution** — `AdaptiveBayesWeighting`: a conjugate‑normal, exponentially
> decayed posterior that tilts capital toward managers with *sustained*
> high scores while letting laggards mean‑revert.

### 12.1  Class contract

```python
class AdaptiveBayesWeighting(BaseWeighting):
    """State‑ful, cross‑period Bayesian re‑weighting.

    Posterior mean := capital share per fund.
    Posterior τ     := confidence (inverse variance) updated via scores.
    """

    def __init__(
        self,
        *,
        half_life: int = 90,          # days – exponential decay of τ
        obs_sigma: float = 0.25,      # σ of score observation noise
        max_w: float | None = 0.20,   # optional hard cap on any single fund
        prior_mean: Literal["equal"] | ndarray = "equal",
        prior_tau: float = 1.0,
    ):
        ...

    def update(
        self,
        scores: pd.Series,            # index = fund, float64
        days: int                     # # calendar days since last rebalance
    ) -> None:
        """Bayes‑update posteriors in‑place (no return)."""

    def weight(self, candidates: pd.DataFrame) -> pd.Series:
        """Return weights **for this period**, sum == 1.0, respects `max_w`."""

12.2  Engine wiring
selector   = build_selector(cfg)          # unchanged
weighting  = build_weighting(cfg)         # may be AdaptiveBayesWeighting

for date, sf in score_frames.items():
    selected = selector.select(sf)[0]     # DataFrame
    weights  = weighting.weight(selected) # uses *current* posterior
    portfolio.rebalance(date, weights)

    # --- NEW: feed realised scores back in ---
    weighting.update(
        scores = sf.loc[weights.index, cfg.rank_column],
        days   = (date - prev_date).days
    )
    prev_date = date

12.3  Config schema delta
portfolio:
  weighting:
    name: adaptive_bayes           # new
    params:
      half_life: 90                # int days
      obs_sigma: 0.25              # score σ
      max_w: 0.20                  # clip (optional)
      prior_tau: 1.0               # prior precision

Back‑compat: absence of these keys defaults to score_prop_simple.

12.4  GUI additions
Weighting method dropdown now enumerates via the plug‑in registry and
auto‑discovers AdaptiveBayesWeighting.

If chosen, reveal controls:

half_life → IntSlider(30‑365)

obs_sigma → FloatSlider(0‑1)

max_w     → FloatSlider(0‑0.5)

prior_tau → FloatSlider(0‑5)

All four controls inherit the 300 ms debounce wrapper.

12.5  State & reset rules
Posterior state lives inside the weighting instance only.

Loading a new YAML (Step 0) or clicking “↻ Reset” in the GUI rebuilds
the weighting object → posteriors reset.

Periodic state can be serialised alongside ~/.trend_gui_state.yml
using pickle under the key adaptive_bayes_posteriors; safe to ignore
if absent.

12.6  Tests (new tests/test_adaptive_bayes.py)
Drift toward winners
Simulate three periods; fund A top‑scores each time.
assert weights_A_3 > weights_A_2 > weights_A_1.

Sum‑to‑one
numpy.testing.assert_allclose(weights.sum(), 1.0, rtol=1e‑12).

Clip respects max_w
Force outlier posterior, assert weights.max() <= max_w + 1e‑9.

Half‑life zero → simple weighting
With half_life = 0, compare against ScorePropSimple.

12.7  Open parameters for Phase 2 sign‑off
Parameter	Default	Rationale	Can be tuned later?
half_life	90 d	echo typical quarterly review cycle	✅
obs_sigma	0.25	reasonable dispersion of z‑scores	✅
max_w	20 %	avoid concentration risk	✅
prior_tau	1.0	uninformative prior	✅


---

**Status**

The `AdaptiveBayesWeighting` implementation and accompanying unit tests are
now merged. The GUI enumerates all weighting plug‑ins automatically and the
defaults listed above match the shipped configuration. Persistent skill can
therefore be modelled out‑of‑the‑box.

## Demo pipeline (maintenance / CI)

Keep the demo scripts in sync with exporter and pipeline changes. Whenever a new
feature lands, run the sequence below and update `config/demo.yml` or
`scripts/run_multi_demo.py` so the demo exercises every code path. See
[../DemoMaintenance.md](../DemoMaintenance.md) for a concise checklist.

1. **Bootstrap the environment**
   ```bash
   ./scripts/setup_env.sh
   ```
2. **Generate the demo dataset**
   ```bash
   python scripts/generate_demo.py [--no-xlsx]
   ```
   The optional flag skips the Excel copy when binary artefacts are not needed.
3. **Run the full demo pipeline and export checks**
   ```bash
   python scripts/run_multi_demo.py
   ```
   The script must call `export.export_data()` so CSV, Excel, JSON **and TXT**
   outputs are produced in one go. Extend the script and config whenever new
   exporter options are introduced.
4. **Run the test suite**
   ```bash
   ./scripts/run_tests.sh
   ```
5. **Keep demo config current**
   - Update `config/demo.yml` and demo scripts whenever export or pipeline
     behaviour changes so the demo covers all features.


