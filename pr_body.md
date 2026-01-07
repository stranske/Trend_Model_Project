<!-- pr-preamble:start -->
> **Source:** Issue #4180

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
An invalid config produced by NL should never reach the analysis pipeline. This safety gate:
- Prevents cryptic runtime errors deep in the pipeline
- Gives users actionable feedback on what's wrong
- Maintains trust that NL changes are safe

#### Tasks
- [x] Create `validate_config(config: dict) -> ValidationResult`:
- [x] - Run existing config validation if any
- [x] - Add additional semantic checks
- [x] - Return structured result with all errors
- [x] Define `ValidationResult` model:
- [x] ```python
- [x] class ValidationError(BaseModel):
- [x] path: str           # e.g., "analysis.top_n"
- [x] message: str        # Human-readable error
- [x] expected: str       # What was expected
- [x] actual: Any         # What was provided
- [x] suggestion: str | None  # How to fix
- [x] class ValidationResult(BaseModel):
- [x] valid: bool
- [x] errors: list[ValidationError]
- [x] warnings: list[ValidationError]
- [x] ```
- [x] Implement semantic validations:
- [x] - Required fields present
- [x] - Types match expected (int, float, str, list, dict)
- [x] - Data source types validated (csv_path, managers_glob)
- [x] - Values within allowed ranges
- [x] - Enum values are valid choices
- [x] - Date ranges make sense (in_start < in_end < out_start < out_end)
- [x] - Cross-field consistency (e.g., top_n <= number of available funds)
- [x] Convert validation errors to user-readable messages:
- [x] - Include the config path that failed
- [x] - State what was expected
- [x] - Show what was provided
- [x] - Suggest a fix when possible
- [x] Integrate with patch application flow:
- [x] ```python
- [x] def apply_and_validate(config: dict, patch: ConfigPatch) -> tuple[dict, ValidationResult]:
- [x] new_config = apply_patch(config, patch)
- [x] result = validate_config(new_config)
- [x] return new_config, result
- [x] ```
- [x] Add "strict mode" that treats warnings as errors

#### Acceptance criteria
- [x] Invalid configs are rejected before reaching pipeline
- [x] Error messages include:
- [x] - Which field failed
- [x] - What was expected
- [x] - What was provided
- [x] - Suggested fix (when determinable)
- [x] CLI shows validation errors clearly
- [x] Streamlit shows validation errors in UI
- [x] Unit tests cover:
- [x] - Missing required fields
- [x] - Wrong types
- [x] - Out-of-range values
- [x] - Invalid enum values
- [x] - Date range violations
- [x] - Cross-field inconsistencies

<!-- auto-status-summary:end -->
