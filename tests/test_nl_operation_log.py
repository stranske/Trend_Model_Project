from __future__ import annotations

from datetime import datetime, timezone

from trend_analysis.config.patch import ConfigPatch, PatchOperation
from trend_analysis.config.validation import ValidationResult
from trend_analysis.llm.nl_logging import NLOperationLog


def test_nl_operation_log_accepts_structured_fields() -> None:
    patch = ConfigPatch(
        operations=[PatchOperation(op="set", path="analysis.top_n", value=12)],
        summary="Update top_n selection",
    )
    validation = ValidationResult(valid=True, errors=[], warnings=[])
    log = NLOperationLog(
        request_id="req-123",
        timestamp=datetime.now(tz=timezone.utc),
        operation="nl_to_patch",
        input_hash="hash-abc",
        prompt_template="template",
        prompt_variables={"instruction": "Set top_n to 12"},
        model_output='{"operations": [], "summary": "noop"}',
        parsed_patch=patch,
        validation_result=validation,
        error=None,
        duration_ms=12.5,
        model_name="unit-test-model",
        temperature=0.0,
        token_usage={"prompt_tokens": 5, "completion_tokens": 7},
    )

    payload = log.model_dump()
    assert payload["request_id"] == "req-123"
    assert payload["operation"] == "nl_to_patch"
    assert payload["parsed_patch"]["summary"] == "Update top_n selection"
