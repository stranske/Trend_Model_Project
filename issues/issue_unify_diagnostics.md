# Unify diagnostics across CLI, Streamlit, and multi-period engine

## Why

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

## Scope

Standardize a single "run payload contract" that all pipeline entry points return, eliminating the need for result coercion.

## Non-Goals

- Removing `DiagnosticResult` (it's the right abstraction)
- Breaking the public `run()` function return type
- Changing how diagnostics are logged/displayed

## Tasks

- [ ] Define canonical `RunPayload` protocol/dataclass with `value`, `diagnostic`, `metadata` fields
- [ ] Update `PipelineResult` to implement `RunPayload` protocol
- [ ] Update `pipeline.run_full()` to always return `RunPayload`-compliant object
- [ ] Update `pipeline.run()` to extract `value` from `RunPayload` (backward compat)
- [ ] Update multi-period engine to expect `RunPayload` instead of duck-typing
- [ ] Remove `_coerce_analysis_result()` once all callers use typed interface
- [ ] Update Streamlit result handling to use `RunPayload` interface
- [ ] Add type guards: `is_run_payload(obj) -> TypeGuard[RunPayload]`
- [ ] Deprecate direct `AnalysisResult` usage with warning

## Acceptance Criteria

- [ ] Single `RunPayload` type is returned by all pipeline entry points
- [ ] Multi-period engine no longer needs `_coerce_analysis_result()`
- [ ] Type checker (mypy) validates payload handling without `Any` casts
- [ ] CLI diagnostic output unchanged
- [ ] Streamlit diagnostic display unchanged  
- [ ] All existing tests pass

## Implementation Notes

Files to modify:
- `src/trend/diagnostics.py` - Add `RunPayload` protocol
- `src/trend_analysis/diagnostics.py` - Update `PipelineResult` 
- `src/trend_analysis/pipeline.py` - Ensure consistent return types
- `src/trend_analysis/multi_period/engine.py` - Remove coercion, use typed interface
- `streamlit_app/` - Update result handling

Proposed `RunPayload` structure:
```python
@dataclass
class RunPayload(Generic[T]):
    value: T | None
    diagnostic: DiagnosticPayload | None
    metadata: dict[str, Any]
    
    def unwrap(self) -> T:
        if self.value is None:
            raise ValueError(self.diagnostic.message if self.diagnostic else "No value")
        return self.value
    
    @property
    def success(self) -> bool:
        return self.value is not None
```

This unifies the existing `DiagnosticResult.value` + `PipelineResult.result` patterns into one contract.
