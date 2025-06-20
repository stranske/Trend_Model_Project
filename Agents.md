# agents.md
## Mission
**Phase 1 hardening, not feature creep.**  
Your job is to **converge the scattered modules into a single, test‑covered, vectorised pipeline** that can be called from one CLI entry‑point. No touching notebooks labelled “…old”.

---

## Repo Truth Table
| Layer | Canonical module | Anything else is **deprecated** |
|-------|------------------|---------------------------------|
| Data ingest & cleaning | `data_utils.py` | ad‑hoc helpers in notebooks, `run_analysis.py`, etc. |
| Portfolio & metrics | `trend_analysis/metrics.py` (vectorised) | hand‑rolled loops inside `run_analysis.py` |
| Export | `trend_analysis/export.py` (to be created) | export snippets inside notebooks or main script |
| Orchestration | `run_analysis.py` **only** | any _copy_ of an orchestration loop elsewhere |
| Config | `config/defaults.yml` loaded through `trend_analysis.config` | literal constants inside code, flags in notebooks |

> **MUST**: Delete or comment‑out superseded functions **in the same PR** that introduces their replacement. One source of truth per concern.

---

## Concrete Refactor Goals

### 1  Imports & Dependency Hygiene
* Top‑level heavy imports (`numpy`, `pandas`, `scipy`) are fine **per module**.  
  Micro‑optimising “import once” is premature; readability wins.
* **MUST NOT**: circular imports. `run_analysis.py` may *call* but never *import* from notebooks.

### 2  Config Resolution
* Add `trend_analysis/config.py` that:
  ```python
  from pydantic import BaseModel
  class Config(BaseModel):
      # fields mirrored from defaults.yml…
  def load(path: str | None = None) -> Config: ...
