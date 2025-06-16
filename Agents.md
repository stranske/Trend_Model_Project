# agents.md
## Do not use the work branch. Only update code on origin main or on the codex-session-fix branch. If something prompts you to move to a different branch, immediately stop work and let me know about the issue.

## Mission
Complete and maintain **Phase 1** of the Vol‑Adjusted Trend Analysis project.  
_No feature work for later phases yet: the current mandate is **refactor, harden, document, and validate** what already exists._

---

## Guiding Principles
1. **Isolation of concerns**  
   *Pure functions for calculations*, separate I/O wrappers, and a thin CLI.  
   Calculation modules must import **nothing** from `matplotlib`, Excel writers, or any other presentation layer.

2. **Config‑driven everything**  
   *No magic numbers.*  
   The YAML file `config/defaults.yml` is the single source for:  
   ```yaml
   data:
     price_files_glob: "data/raw/*.csv"
     index_files_glob:  "data/indices/*.csv"
   in_sample:
     split:            "2017-12-31"     # or "0.70" for 70 %
   risk_free_rate_annual: 0.02
   target_vol:         0.10
   portfolio:
     selection:        "all"            # all | random | manual
     random_seed:      42
   output:
     formats:          ["xlsx", "csv", "json"]
     directory:        "results/"

Vectorisation first, loops last
• Use numpy/pandas native ops, np.where, np.select, and pd.eval before for loops.
• One‑pass calculations for drawdowns and running vol (see metrics.py).
• Accept Series/DataFrame plus optional axis param; return the same shape.

Reproducibility & determinism
• Set the global NumPy random seed from config as early as possible.
• Use only deterministic pandas resampling (no implicit time‑zone localisations).

Typing + style gates
Every public function must have:

PEP 484 type hints

NumPy‑style docstring with Args / Returns / Raises / Example

Unit tests in tests/ hitting ≥ 90 % coverage.
CI fails on: ruff, black --check, mypy, pytest.

Zero‑state friendly
The repo should clone and run with make demo ↦ creates a dummy dataset, runs Phase 1, and writes artefacts to results/.

trend_model_project/
│
├── config/                # yaml + schema
│   └── defaults.yml
│
├── trend_analysis/        # importable package
│   ├── __init__.py
│   ├── io.py              # load_*(), save_*()
│   ├── preprocessing.py    # cleaning / alignment helpers
│   ├── metrics.py          # CAGR, vol, Sharpe, Sortino, DD (vectorised)
│   ├── portfolio.py        # selection + aggregation
│   ├── analyse.py          # orchestrates in/out‑sample run
│   └── cli.py             # thin Typer or argparse entry‑point
│
├── notebooks/             # exploratory only – non‑blocking
│
├── tests/
│   └── test_*.py
│
├── results/               # git‑ignored
│
├── agents.md              # ← (this file)
└── README.md

Task Backlog (Phase 1 only)
Priority	Title	Acceptance Criteria
P0	Extract metrics to metrics.py	Functions fully vectorised, 100 % tested, pass stochastic spot‑checks vs. existing notebook output
P0	Replace hard‑coded params with YAML	Notebook and scripts must not reference literal dates, rf‑rate, or target vol
P1	Implement portfolio.py selection logic	Supports "all", "random", "manual"; deterministic under seed
P1	Add CLI	python -m trend_analysis.run --config config/defaults.yml runs whole phase
P2	CI pipeline	GitHub Actions workflow running lint/type/test on push & PR
P2	Demo dataset + make target	make demo completes end‑to‑end in <30 s

Coding Dos & Don’ts
DO keep state immutable where reasonable → return new DataFrames, don’t mutate in place.

DO raise ValueError with actionable messages when inputs fail checks.

DON’T bury DataFrame index assumptions – use explicit column names.

DON’T commit data ≥ 1 MB or any secrets.

Definition of Done (Phase 1)
One‑command run produces:

results/performance_in_sample.{xlsx,csv,json}

results/performance_out_sample.{xlsx,csv,json}

CI green; code coverage ≥ 90 %.

Changelog entry in CHANGELOG.md with date & semver bump (v0.1.0 → v0.2.0).

README quick‑start reflects final CLI invocation.
