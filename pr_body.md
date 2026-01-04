<!-- pr-preamble:start -->
> **Source:** Issue #4147

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
There are multiple diagnostic/result wrappers in use:
1. `DiagnosticResult` in `src/trend/diagnostics.py` - Generic wrapper with value + diagnostic payload
2. `PipelineResult` in `src/trend_analysis/diagnostics.py` - Pipeline-specific success/failure
3. `AnalysisResult` - Legacy wrapper for backward compatibility
4. Multi-period engine's `_coerce_analysis_result()` in `src/trend_analysis/multi_period/engine.py` - Normalizes between formats

The multi-period engine has to guess whether results are dicts, wrappers, or legacy objects:
```python
def _coerce_analysis_result(result: object) -> tuple[dict[str, Any] | None, DiagnosticPayload | None]:
    """Normalise pipeline outputs regardless of legacy or diagnostic wrappers."""
    diag = getattr(result, "diagnostic", None)
    if hasattr(result, "unwrap"):
        unwrap = getattr(result, "unwrap", None)
```

This "duck-typing to figure out what we got back" pattern is fragile and error-prone.

#### Tasks
- [ ] Define canonical `RunPayload` protocol/dataclass with `value`, `diagnostic`, `metadata` fields
- [x] Update `PipelineResult` to implement `RunPayload` protocol
- [ ] Update `pipeline.run_full()` to always return `RunPayload`-compliant object
- [ ] Update `pipeline.run()` to extract `value` from `RunPayload` (backward compat)
- [ ] Update multi-period engine to expect `RunPayload` instead of duck-typing
- [ ] Remove `_coerce_analysis_result()` once all callers use typed interface
- [ ] Update Streamlit result handling to use `RunPayload` interface
- [ ] Add type guards: `is_run_payload(obj) -> TypeGuard[RunPayload]`
- [ ] Deprecate direct `AnalysisResult` usage with warning

#### Acceptance criteria
- [ ] Single `RunPayload` type is returned by all pipeline entry points
- [ ] Multi-period engine no longer needs `_coerce_analysis_result()`
- [ ] Type checker (mypy) validates payload handling without `Any` casts
- [ ] CLI diagnostic output unchanged
- [ ] Streamlit diagnostic display unchanged
- [ ] All existing tests pass

<!-- auto-status-summary:end -->
