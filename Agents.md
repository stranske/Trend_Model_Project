# agents.md
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

