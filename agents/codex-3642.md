<!-- bootstrap for codex on issue #3642 -->

## Scope
- Decouple multi-period compute steps from CSV file I/O by adding dedicated loaders plus an orchestration entry point so tests and reusable callers can pass in-memory DataFrames.

## Current status snapshot
The following blocks **must** be copied verbatim into each keepalive update comment. Only check off a task when the acceptance criteria below are satisfied, and re-post the entire set whenever a box flips state.

### Tasks
- [x] Introduce `load_prices`, `load_membership`, and `load_benchmarks` functions that return DataFrames with validated schemas.
- [x] Refactor the pipeline so downstream steps accept DataFrames, not file paths.
- [x] Add a single orchestration function that wires I/O to compute steps.

### Acceptance criteria
- [x] Core compute functions have no file I/O side effects.
- [x] Swapping in-memory fixtures for CSVs in tests is trivial.

## Progress log
- 2025-11-18 – Captured the scope and keepalive checklist so nudges can continue; all loader/orchestration tasks and acceptance criteria are complete per the latest implementation and tests (`pytest tests/test_multi_period_engine_keepalive.py tests/test_multi_period_loaders.py`).
