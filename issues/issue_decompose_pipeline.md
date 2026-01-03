# Decompose pipeline.py into focused stage modules

## Why

`pipeline.py` is 3003 lines and serves as orchestrator, validator, selector, weight-engine integrator, risk controller, diagnostics builder, and more. While decomposition has started (stage dataclasses, helper functions), the module still violates single-responsibility and makes it difficult to:

1. Test individual stages in isolation
2. Understand the data flow without reading thousands of lines
3. Modify one concern without risking regressions in others
4. Onboard new contributors

The module handles: preprocessing (calendar + missing policy + inception mask), selection/scoring, and portfolio construction (weights + constraints + risk)—three distinct conceptual domains that should be separate modules.

## Scope

Split pipeline.py into 3 focused modules while keeping `pipeline.run()` as a thin conductor. This is modular decomposition, not microservices.

## Non-Goals

- Creating new public APIs
- Changing the `run()` function signature or return type
- Breaking existing imports (`from trend_analysis.pipeline import run`)
- Introducing dependency injection frameworks

## Tasks

- [ ] Create `src/trend_analysis/stages/__init__.py` package
- [ ] Extract preprocessing logic into `src/trend_analysis/stages/preprocessing.py`:
  - Calendar alignment (`align_calendar`)
  - Missing data policy (`apply_missing_policy`, `MissingPolicyResult`)
  - Inception masking logic
  - Frequency detection and normalization
- [ ] Extract selection/scoring logic into `src/trend_analysis/stages/selection.py`:
  - Fund ranking (`rank_select_funds`)
  - Score computation
  - Metric bundle computation
  - Risk stats configuration
- [ ] Extract portfolio construction into `src/trend_analysis/stages/portfolio.py`:
  - Weight computation and constraints
  - Risk adjustments
  - Vol targeting
  - Final weight application
- [ ] Refactor `pipeline.py` to import from stages and orchestrate
- [ ] Keep `run()`, `run_full()`, `run_analysis()` in pipeline.py as thin conductors
- [ ] Maintain backward-compatible imports via `__all__` in pipeline.py
- [ ] Add integration test verifying stage isolation doesn't change outputs

## Acceptance Criteria

- [ ] `pipeline.py` reduced to <500 lines (orchestration + public API)
- [ ] Each stage module is independently testable
- [ ] `from trend_analysis.pipeline import run` continues to work
- [ ] All existing pipeline tests pass without modification
- [ ] No circular imports between stages
- [ ] New stage modules have >80% test coverage

## Implementation Notes

Directory structure:
```
src/trend_analysis/
├── pipeline.py           # Thin orchestrator (<500 lines)
└── stages/
    ├── __init__.py       # Re-export stage functions
    ├── preprocessing.py  # ~400 lines
    ├── selection.py      # ~600 lines  
    └── portfolio.py      # ~500 lines
```

Key functions to move:
- `_preprocessing_summary()`, `_build_inception_mask()` → preprocessing
- `_compute_stats()`, `rank_select_funds()` → selection
- `_compute_portfolio_weights()`, `compute_constrained_weights()` → portfolio

`pipeline.run()` becomes:
```python
def run(cfg: Config) -> pd.DataFrame:
    data = preprocessing.load_and_clean(cfg)
    selected = selection.select_funds(data, cfg)
    portfolio = portfolio.construct(selected, cfg)
    return portfolio.to_dataframe()
```
