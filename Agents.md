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
   # ======================================================================
# Vol‑Adjusted Trend Analysis ― Phase‑1 master config
# ======================================================================
version: "0.1.0"                 # bump when the schema changes

# ----------------------------------------------------------------------
# 1. DATA INGESTION
# ----------------------------------------------------------------------
data:
  managers_glob:      "data/raw/managers/*.csv"
  indices_glob:       "data/raw/indices/*.csv"
  date_column:        "Date"          # ISO‑8601 expected
  price_column:       "Adj_Close"     # or "NAV", etc.
  frequency:          "D"             # D | W | M
  timezone:           "UTC"           # for parsing
  currency:           "USD"
  nan_policy:         "ffill"         # drop | ffill | bfill | both
  lookback_required:  756             # minimum obs (≈3 yrs daily)

# ----------------------------------------------------------------------
# 2. PRE‑PROCESSING OPTIONS
# ----------------------------------------------------------------------
preprocessing:
  de_duplicate:       true
  winsorise:
    enabled:          true
    limits:           [0.01, 0.99]    # two‑sided
  log_prices:         false           # turn on if prices span decades
  holiday_calendar:   "NYSE"          # forward fill non‑trading days
  resample:
    target:           "D"             # keep None to skip
    method:           "last"          # last | mean | sum | pad
    business_only:    true

# ----------------------------------------------------------------------
# 3. VOLATILITY ADJUSTMENT
# ----------------------------------------------------------------------
vol_adjust:
  enabled:            true
  target_vol:         0.10            # annualised (10 %)
  window:
    length:           63             # rolling window (≈3 months)
    decay:            "ewma"          # ewma | simple
    lambda:           0.94           # EWMA decay
  floor_vol:          0.04            # avoid 100× leverage

# ----------------------------------------------------------------------
# 4. IN‑SAMPLE / OUT‑OF‑SAMPLE SPLIT
# ----------------------------------------------------------------------
sample_split:
  method:             "date"          # date | ratio
  date:               "2017-12-31"    # ignored if method=ratio
  ratio:              0.70            # ignored if method=date
  rolling_walk:       false           # future‑phase feature flag
  folds:              5               # for CV, if rolling_walk=true

# ----------------------------------------------------------------------
# 5. PORTFOLIO CONSTRUCTION
# ----------------------------------------------------------------------
portfolio:
  selection_mode:     "all"           # all | random | manual
  random_seed:        42
  random_n:           10              # num managers if random
  manual_list:        []              # filled when selection_mode=manual
  weighting_scheme:   "equal"         # equal | vol_inverse | custom
  rebalance_freq:     "M"             # M | Q | A | None
  leverage_cap:       2.0             # max gross exposure

# ----------------------------------------------------------------------
# 6. PERFORMANCE METRICS
# ----------------------------------------------------------------------
metrics:
  rf_rate_annual:     0.02            # flat rf or link to series
  use_continuous:     false           # geometric (default) vs. log
  alpha_reference:    "SP500TR"       # ticker in indices_glob
  compute:
    - CAGR
    - Volatility
    - Sharpe
    - Sortino
    - Max_Drawdown
    - Calmar
    - Hit_Rate
    - Skew
  bootstrap_ci:
    enabled:          true
    n_iter:           2000
    ci_level:         0.95

# ----------------------------------------------------------------------
# 7. EXPORT OPTIONS
# ----------------------------------------------------------------------
export:
  directory:          "results/"
  formats:            ["xlsx", "csv", "json"]
  excel:
    autofit_columns:  true
    number_format:    "0.00%"
    conditional_bands:
      enabled:        true
      palette:        "RdYlGn"
  include_raw_returns: true
  include_vol_adj:     true
  include_figures:     false           # Phase‑2

# ----------------------------------------------------------------------
# 8. LOGGING & EXECUTION
# ----------------------------------------------------------------------
run:
  log_level:          "INFO"          # DEBUG | INFO | WARNING | ERROR
  log_file:           "logs/phase1.log"
  n_jobs:             -1              # ‑1 ⇒ all CPUs
  cache_dir:          ".cache/"
  deterministic:      true            # sets NumPy & pandas options


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
