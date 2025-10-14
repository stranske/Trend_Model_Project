# Agents.md
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

## Automation entry points (Orchestrator → bridge → verification)

### Agents 70 Orchestrator
- **File**: `.github/workflows/agents-70-orchestrator.yml`.
- **Triggers**: 20-minute cron plus on-demand `workflow_dispatch` inputs for readiness, bootstrap, verification, watchdog, and keepalive paths.
- **Role**: single automation front door. Every scheduled sweep and manual run invokes `reusable-16-agents.yml`, passing the merged toggles (watchdog enabled by default, verification toggled via `enable_verify_issue`).
- **Manual run**: Actions → **Agents 70 Orchestrator** → **Run workflow**. Supply booleans as strings (for example `true`) and optional JSON overrides in `options_json` when you need to bundle advanced switches in one payload.

### Agents 63 Codex Issue Bridge
- **File**: `.github/workflows/agents-63-codex-issue-bridge.yml`.
- **Triggers**: reacts to issue events (opened, labeled, reopened) when the issue carries the `agent:codex`/`agents:codex` label set, and exposes `workflow_dispatch` inputs for manual testing, PR mode overrides, and forcing draft PRs.
- **Role**: hydrates bootstrap PRs or invite flows for labeled issues, optionally posting the `@codex start` command. Manual dispatch requires `test_issue` so the bridge knows which issue to service.

### Agents 64 Verify Agent Assignment
- **File**: `.github/workflows/agents-64-verify-agent-assignment.yml`.
- **Triggers**: reusable `workflow_call` with a required `issue_number` plus manual `workflow_dispatch` for spot checks.
- **Role**: confirms that labeled issues still carry an agent assignee and emits a Markdown table + status outputs consumed by the orchestrator. Manual runs follow the same inputs—enter the issue number and review the generated run summary.

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

### 2025-09-18 UPDATE — SCALAR METRIC MEMOIZATION (ISSUE #1156)

An opt-in memoization layer now caches scalar per‑fund metric series accessed via `WindowMetricBundle.ensure_metric`. Enable with:

```yaml
performance:
  cache:
    metrics: true
```

Default remains off (non‑breaking). Covariance payload caching is unaffected. Refer to `docs/metric_cache.md` for details.

### 2025-09-18 UPDATE — PR DRAFT TOGGLE (ISSUE #1137)

Codex bootstrap workflow now supports a `codex_pr_draft` input (workflow_dispatch) controlling whether the automatically created PR is a draft. Default remains non‑draft to ensure agent engagement (addressing Issue #1137). Fallback and composite bootstrap paths both consume the resolved draft flag so behavior is consistent. Invite mode (issue-triggered path) still requires a human-authored PR and is unaffected.

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
- Always run `black` on changed files before committing so lint passes.

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

### 2025‑07‑13 UPDATE — MULTI‑PERIOD EXPORT ROADMAP

The backlog still includes emitting a Phase‑1 style workbook with **one tab per
period** for multi‑period runs.  CSV and JSON outputs must create one file per
period using the same summary table produced by
``summary_frame_from_result``.  A ``summary`` sheet/file aggregates portfolio
returns across all periods, formatted identically to each individual period
sheet.  Work is in progress to expose this via a new ``run_multi_analysis`` CLI
that calls ``export.export_multi_period_metrics``.

### 2025-07-20 UPDATE — MULTI-PERIOD OUTPUT SPEC

The pending export work shall produce an Excel workbook with one worksheet per period and a final `summary` sheet. Each tab uses the exact Phase‑ style summary table via `make_period_formatter`. CSV and JSON exports mirror this by writing one file per period plus a `_summary` file. A new helper `period_frames_from_results()` converts a sequence of result dictionaries into the mapping consumed by `export_multi_period_metrics`.

### 2025-08-01 UPDATE — PER-PERIOD METRICS WORKBOOK

Phase‑2 back-tests shall emit an Excel workbook with one tab per period formatted identically to the Phase‑1 summary sheet. CSV and JSON outputs produce one file per period in the same table form. In addition, a `summary` tab (and `_summary` file) aggregates portfolio returns across all periods using the identical layout. Implementation is underway in `export.export_multi_period_metrics`.

### 2025-08-10 UPDATE — MULTI-PERIOD PHASE‑1 METRICS EXPORT

Finalise the design goal that each period of a rolling back‑test mirrors the
Phase‑1 metrics sheet. The exporter shall generate a workbook with one tab per
period and a `summary` tab combining portfolio returns. CSV and JSON formats
receive one file per period plus a `_summary` file. Implementation has started in
`export.workbook_frames_from_results` and the updated
`export.export_multi_period_metrics` helper.

### 2025-08-20 UPDATE — CONSOLIDATED CSV/JSON OUTPUT

The design goal remains one Excel worksheet per period plus a `summary` tab.
CSV and JSON exports now consolidate all period tables into **one** file named
`*_periods.*` with an accompanying `*_summary.*` file holding the aggregated
portfolio returns.  Raw metrics, when requested, follow the same pattern under
`metrics.*` and `metrics_summary.*`.

### 2025-08-30 UPDATE — PER-PERIOD PHASE‑1 WORKBOOK

Implementation work continues to realise the original goal: multi‑period runs
should emit an Excel workbook with **one tab per period** using the exact
Phase‑1 summary layout.  A final `summary` sheet aggregates portfolio returns
across all periods in the same format.  CSV and JSON outputs mirror this by
bundling all period tables into a single `*_periods.*` file alongside a
`*_summary.*` file.  Helper `export_phase1_workbook()` now begins the build
out of this feature.

### 2025-09-05 UPDATE — PHASE‑1 MULTI-PERIOD OUTPUT

The design intent remains unchanged: a rolling back‑test should deliver an
Excel workbook with one tab per period plus a final `summary` tab combining
portfolio results across all periods. CSV and JSON formats must consolidate the
same tables into a single `*_periods.*` file and a companion `*_summary.*`
file. Work continues in `export_phase1_workbook()` and the revised
`export_multi_period_metrics` function.

### 2025-09-15 UPDATE — PHASE‑1 METRICS GOAL

The original requirement is still outstanding: metrics from each period of a
multi-period run should appear in an Excel workbook with one sheet per period
using the **exact** Phase‑1 formatting, plus a `summary` sheet combining the
portfolio across periods. CSV and JSON outputs must present the same tables as
a single `*_periods.*` file with a matching `*_summary.*` file. Implementation
has begun but is not yet complete.


### 2025-09-30 UPDATE — MULTI-PERIOD WORKBOOK TARGET

Development continues on the export layer. The goal remains to emit a Phase‑1
style workbook with **one sheet per period** and a final `summary` sheet
combining portfolio returns across all periods. Each sheet must apply the same
formatters as the original single-period export. CSV and JSON outputs will
bundle all period tables into a single `*_periods.*` file with a companion
`*_summary.*` file. The new helpers `export_phase1_workbook()` and
`export_phase1_multi_metrics()` begin this implementation.

### 2025-10-05 UPDATE — MULTI-PERIOD PHASE-1 EXPORT PROGRESS

`export_phase1_workbook()` now builds its sheet mapping via
`workbook_frames_from_results()` so each period tab and the combined
`summary` tab share the exact Phase‑1 layout. `export_phase1_multi_metrics()`
uses the same helper to emit a single ``*_periods.*`` file plus a matching
``*_summary.*`` for CSV/JSON users. Development continues to keep these
outputs identical to the single-period workbook.

### 2025-10-12 UPDATE — MULTI-PERIOD PHASE-1 METRICS GOAL

The export layer must emit a Phase-1 style workbook with one sheet per period and a final `summary` sheet combining portfolio returns. Each sheet uses `make_period_formatter` so formatting matches the single-period output. CSV and JSON outputs consolidate these tables into a single `*_periods.*` file with a companion `*_summary.*` file. Implementation now begins in `export_phase1_workbook()` and `export_phase1_multi_metrics()` to realise this design.

### 2025-10-20 UPDATE — PHASE-1 WORKBOOK TARGET

Multi-period exports shall produce an Excel workbook with one sheet per period plus a final `summary` sheet combining portfolio returns. Formatting and columns must match the Phase-1 output. CSV and JSON formats consolidate all period tables into a single `*_periods.*` file with a corresponding `*_summary.*` file. Development now adds `flat_frames_from_results()` to build these consolidated tables for the exporters.

### 2025-10-31 UPDATE — MULTI-PERIOD PHASE-1 EXPORT DESIGN

The Phase-1 metrics export shall produce one Excel worksheet per period using the exact same formatting as the single-period report. A final `summary` tab aggregates portfolio performance across all periods in the identical layout. CSV and JSON exports consolidate all period tables into a single `*_periods.*` file and place the combined results in a matching `*_summary.*` file. Implementation starts in `export_phase1_workbook()` and `export_phase1_multi_metrics()` which build their frames via `workbook_frames_from_results()`.

### 2025-11-08 UPDATE — PHASE-1 MULTI-PERIOD WORKBOOK GOAL

The outstanding goal is unchanged: each multi-period run should yield a Phase‑1 style workbook with **one sheet per period** and a final `summary` sheet aggregating portfolio returns. The summary tab must mirror the columns and formatting of the individual period sheets. CSV and JSON outputs bundle all period tables into a single `*_periods.*` file alongside a matching `*_summary.*` file. Current work focuses on finishing `export_phase1_workbook()` and surfacing these helpers via the public API.

### 2025-11-22 UPDATE — MULTI-PERIOD METRICS EXPORT TARGET

The original Phase‑1 metrics layout shall extend to multi‑period runs. The exporter must generate an Excel workbook with one tab per period and a final `summary` tab aggregating portfolio returns in the **exact** same column order and formatting. CSV and JSON outputs deliver the period tables in a single `*_periods.*` file and the combined returns in a matching `*_summary.*` file. Implementation has begun in the export helpers; keep working toward full CLI integration and parity across formats.


### 2025-12-01 UPDATE — PHASE-1 MULTI-PERIOD OUTPUT SPEC

The original design has not yet been fully realised. Each rolling run should
produce an Excel workbook with one sheet per period, formatted exactly like the
Phase-1 summary table. A final `summary` sheet must combine portfolio returns
with the same columns and formatting. CSV and JSON exports shall mirror this by
bundling all period tables into a single `<prefix>_periods.*` file together with
`<prefix>_summary.*` for the aggregated results. Implementation work continues in
`export_phase1_workbook()` and `export_phase1_multi_metrics()`.

### 2025-12-08 UPDATE — MULTI-PERIOD PHASE-1 EXPORT INITIATIVE

The long-standing objective is for rolling runs to emit a Phase‑1 style
workbook with **one tab per period** and a final `summary` sheet combining
portfolio returns.  Each sheet must share the identical columns and
formatting.  CSV and JSON outputs should deliver a single `<prefix>_periods.*`
file plus `<prefix>_summary.*` to aggregate the results.  Implementation work
begins in `export_phase1_workbook()` and `flat_frames_from_results()` to shape
these consolidated tables.

### 2025-12-15 UPDATE — PHASE-1 MULTI-PERIOD WORKBOOK GOAL

The original requirement remains: metrics from each period must be exported as
an Excel workbook with **one sheet per period** using the exact Phase‑1 summary
formatting. A final `summary` sheet combines portfolio returns across all
periods in the same layout. CSV and JSON outputs should consolidate all period
tables into a single `<prefix>_periods.*` file with a matching
`<prefix>_summary.*` file. Implementation continues in
`export_phase1_workbook()` and `export_phase1_multi_metrics()`.

### 2025-12-22 UPDATE — PHASE-1 EXPORT ROADMAP

Implementation is ongoing to deliver a multi-period Excel workbook with one sheet per period and a final `summary` sheet of combined portfolio returns. Each sheet must use the exact Phase‑1 formatting. CSV and JSON exports shall output a single `<prefix>_periods.*` file and a companion `<prefix>_summary.*` file. Work continues in `export_phase1_workbook()` and `export_phase1_multi_metrics()`.

### 2026-01-05 UPDATE — MULTI-PERIOD OUTPUT OBJECTIVE

The project still needs a full Phase‑1 workbook for rolling runs.  Every period
should populate its own tab with identical formatting, and a final `summary`
tab must aggregate portfolio returns in the very same layout.  CSV and JSON
exports must bundle the per-period tables into one `<prefix>_periods.*` file
with a matching `<prefix>_summary.*` companion file.  Implementation will finish
linking these helpers into the public CLI.

### 2026-01-15 UPDATE — WORKBOOK DATA HELPER

Phase‑1 metrics should export one sheet per period plus a summary tab.  CSV and
JSON outputs must consolidate all periods into a single file.  A new helper
`phase1_workbook_data()` starts this implementation by returning the ordered
sheet mapping for a multi‑period workbook, optionally including raw metrics per
period and for the combined summary.  `export_phase1_workbook()` now builds its
workbook from this mapping.

## Fund Selection Debugging Protocol

### Problem Context
When debugging multi-period portfolio analysis where the same managers are selected every period despite changing performance rankings, follow this systematic approach.

### Debugging Workflow

1. **Environment Setup**
   - Ensure virtual environment is activated: `source venv/bin/activate`
   - Verify all dependencies installed: `pip install -r requirements.txt && pip install -e .`
   - Confirm working on correct branch (use `chore/demo-pipeline` for debugging)

2. **Data Completeness Analysis**
   ```python
   # Run the debug script to check data availability
   python debug_fund_selection.py
   ```

This script will reveal:

- Which managers get filtered out due to missing data in in-sample periods
 
- Which managers get filtered out due to missing data in out-of-sample periods
- Final available manager pool for selection
- Actual ranking results for available managers

Expected Issues to Check

- Data Gap Issue: Only 8 managers have complete data across all periods
- Ranking Bug: Selection logic not sorting by performance metrics
- Configuration Issue: Wrong parameters passed to rank_select_funds
- Period Definition: Incorrect date parsing or period boundaries

Workflow Compliance

- DO NOT make ad-hoc changes to core modules from demo branch
- DO document findings clearly before proposing fixes
- DO follow the phase2-dev → chore/demo-pipeline workflow for fixes
- DO NOT merge anything with main branch

Core Module Fix Process
If debugging reveals bugs in core selection logic:

- Document the Issue
  - Specific function with the bug (e.g., rank_select_funds)
  - Expected vs actual behavior
  - Root cause analysis
  - Test case demonstrating the problem
- Implement Fix on phase2-dev

Common Pitfalls to Avoid

- DON'T assume the ranking algorithm is wrong without checking data completeness first
- DON'T make changes to core modules without switching to phase2-dev
- DON'T run analysis commands that take more than 2-3 minutes without progress updates
- DON'T ignore the virtual environment setup - module imports will fail
- DO trace through the actual data filtering pipeline step by step

## Development Workflow

### Multi-Period Analysis Debugging

When debugging multi-period issues:

1. **Start with data completeness check** - Most "selection not changing" issues are due to insufficient data for additional managers
2. **Use the debug_fund_selection.py script** - Provides systematic analysis of the selection pipeline
3. **Check both in-sample AND out-of-sample data requirements** - Both periods must have complete data for a manager to be eligible
4. **Verify configuration parameters** - Ensure rank_select_funds is getting correct inclusion_approach, n, score_by parameters

Example debug workflow:
```bash
# 1. Run systematic debugging
python debug_fund_selection.py

# 2. If data issue found: investigate data generation
# 3. If logic issue found: follow core fix workflow
# 4. If config issue found: update configuration files
```

## Code Quality Guidelines

### Debugging Script Standards

- Debugging scripts should be self-contained and clearly document their purpose
- Include comprehensive output showing each step of the analysis
- Distinguish between data issues vs. logic bugs vs. configuration problems
- Provide clear conclusions and next steps based on findings
- Follow the same code quality standards as production code

### Rank Selection Bug Fix Protocol

**Problem**: The `rank_select_funds` function in `src/trend_analysis/core/rank_selection.py` selects funds by DataFrame column order instead of by performance ranking.

**Root Cause**: The function calls `scores.head(n)` without sorting the scores first, so it returns the first N funds in the original DataFrame order rather than the top N performers.

**Workflow**:
1. Switch to `phase2-dev` branch
2. Fix the core ranking logic
3. Add comprehensive tests
4. Commit and push to `phase2-dev`
5. Switch back to `chore/demo-pipeline` and merge the fix

**Critical Requirements**:
- DO NOT modify core modules from the demo branch
- All core fixes must be done on `phase2-dev`
- Ensure virtual environment is active before making changes
- Test both ascending (MaxDrawdown) and descending (Sharpe) metrics

