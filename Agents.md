# agents.md
YOU ARE CODEX.  VOL_ADJ_TREND_ANALYSIS – AUTHORITATIVE GUIDE
============================================================

State of the union
------------------
✅ **Phase 1 is COMPLETE and passing CI (tests, 100 % coverage, ruff/black/mypy).**  
   *The code under this section is **frozen** — touch only to fix a red build.*

* Rank‑based manager‑selection, blended z‑score logic, ASCENDING_METRICS guard.
* Risk‑metrics export layer with two‑sheet smoke test & column‑order regression.
* Canonical‑locations refactor; `pipeline.run(cfg)` is pure.

------------------------------------------------------------------------
PHASE‑1 CODE – READ‑ONLY UNLESS TESTS FAIL
------------------------------------------------------------------------

(Full Phase‑1 spec retained here for reference ‑‑ snip ‑‑)

------------------------------------------------------------------------
PHASE‑2 – ***YOU ARE HERE***   Build the scaffolding ONLY
------------------------------------------------------------------------

Directory tree to create (exact spelling):

trend_analysis/
├── pipeline.py # expose single_period_run (was _run_analysis)
├── multi_period/
│ ├── init.py
│ ├── engine.py # stub with raise NotImplementedError
│ ├── scheduler.py # stub
│ └── replacer.py # stub
├── gui/
│ └── app.py # empty Streamlit placeholder
config/
defaults.yml # extend with: multi_period:, jobs:, checkpoint_dir:, random_seed:


### Rename task
* Inside `trend_analysis/pipeline.py` rename `def _run_analysis(...)`
  **→** `def single_period_run(...)` and update all internal calls.

### New YAML keys (place‑holders only)

```yaml
multi_period:
  frequency: "M"      # M | Q | A
  start: "2000-01-01"
  end:   "2025-06-30"
  oos_window: 252     # trading days
  triggers: {}
jobs: 1
checkpoint_dir: null
random_seed: 42

