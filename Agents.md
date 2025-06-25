# agents.md
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
    Step 1  – Mode (‘all’, ‘random’, ‘manual’, **‘rank’**),
               checkboxes for “vol‑adj” and “use ranking”.  
    Step 2  – If mode == 'rank' **or** user ticked “use ranking”
               → reveal controls for `inclusion_approach`,
               `score_by`, `N / Pct / Threshold`, and (if blended)
               three sliders for weights + metric pickers.  
    Step 3  – If mode == 'manual'  
               → display an interactive DataFrame of the IS scores so the
               user can override selection and set weights.
    Step 4  – Output format picker (csv / xlsx / json) then fire
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
🔄 2025‑06‑15 UPDATE — PHASE‑1 ENHANCEMENTS
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

## | Layer / concern                      | **Canonical location**                                                     | Everything else is **deprecated**                         |
| ------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Data ingest & cleaning**           | `trend_analysis/data.py` <br> (alias exported as `trend_analysis.data`)    | `data_utils.py`, helper code in notebooks or `scripts/`   |
| **Portfolio logic & metrics**        | `trend_analysis/metrics.py` (vectorised)                                   | loops inside `run_analysis.py`, ad‑hoc calcs in notebooks |
| **Export / I/O**                     | `trend_analysis/export.py`                                                 | the root‑level `exports.py`, snippets inside notebooks    |
| **Domain kernels (fast primitives)** | `trend_analysis/core/` package                                             | stand‑alone modules under the top‑level `core/` directory |
| **Pipeline orchestration**           | `trend_analysis/pipeline.py` (pure)                                        | any duplicated control flow elsewhere                     |
| **CLI entry‑point**                  | `run_analysis.py` **only** (thin wrapper around `trend_analysis.cli:main`) | bespoke `scripts/*.py` entry points                       |
| **Config**                           | `config/defaults.yml` loaded through `trend_analysis.config.load()`        | hard‑coded constants, magic numbers in notebooks          |
| **Tests**                            | `tests/` (pytest; 100 % branch‑aware coverage gate)                        |    —                                                      |
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
# 🔬 scratchpad – may be deleted at any time.

CI (GitHub Actions) stages to add:

lint  (ruff + black –‑check)

type‑check (mypy, strict)

test (pytest ‑‑cov trend_analysis ‑‑cov‑branch)

build‑wheel (tags only)

