<!-- pr-preamble:start -->
> **Source:** Issue #4145

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
`pipeline.py` is 3003 lines and serves as orchestrator, validator, selector, weight-engine integrator, risk controller, diagnostics builder, and more. While decomposition has started (stage dataclasses, helper functions), the module still violates single-responsibility and makes it difficult to:

1. Test individual stages in isolation
2. Understand the data flow without reading thousands of lines
3. Modify one concern without risking regressions in others
4. Onboard new contributors

The module handles: preprocessing (calendar + missing policy + inception mask), selection/scoring, and portfolio construction (weights + constraints + risk)—three distinct conceptual domains that should be separate modules.

#### Tasks
- [x] Create `src/trend_analysis/stages/__init__.py` package
- [x] Extract preprocessing logic into `src/trend_analysis/stages/preprocessing.py`:
- [x] - Calendar alignment (`align_calendar`)
- [x] - Missing data policy (`apply_missing_policy`, `MissingPolicyResult`)
- [x] - Inception masking logic
- [x] - Frequency detection and normalization
- [x] Extract selection/scoring logic into `src/trend_analysis/stages/selection.py`:
- [x] - Fund ranking (`rank_select_funds`)
- [x] - Score computation
- [x] - Metric bundle computation
- [x] - Risk stats configuration
- [x] Extract portfolio construction into `src/trend_analysis/stages/portfolio.py`:
- [x] - Weight computation and constraints
- [x] - Risk adjustments
- [x] - Vol targeting
- [x] - Final weight application
- [x] Refactor `pipeline.py` to import from stages and orchestrate
- [x] Keep `run()`, `run_full()`, `run_analysis()` in pipeline.py as thin conductors
- [x] Maintain backward-compatible imports via `__all__` in pipeline.py
- [x] Add integration test verifying stage isolation doesn't change outputs

#### Acceptance criteria
- [x] `pipeline.py` reduced to <500 lines (orchestration + public API) — **Currently 400 lines**
- [x] Each stage module is independently testable
- [x] `from trend_analysis.pipeline import run` continues to work
- [x] All existing pipeline tests pass without modification
- [x] No circular imports between stages
- [ ] New stage modules have >80% test coverage — **Coverage verification pending CI**

<!-- auto-status-summary:end -->
