import pytest

from trend.diagnostics import DiagnosticPayload, DiagnosticResult
from trend_analysis.diagnostics import (
    PipelineReasonCode,
    PipelineResult,
    coerce_pipeline_result,
    pipeline_failure,
    pipeline_success,
)


class _MappingWrapper(dict):
    """Mapping subclass to verify conversion to a plain dict."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_pipeline_result_behaves_like_mapping():
    payload = {"alpha": 1, "beta": 2}

    result = pipeline_success(payload)

    assert isinstance(result, PipelineResult)
    assert result["alpha"] == 1
    assert list(result.keys()) == ["alpha", "beta"]
    assert list(result.values()) == [1, 2]
    assert list(result.items()) == [("alpha", 1), ("beta", 2)]
    assert result.get("missing", "fallback") == "fallback"
    assert result.copy() == payload
    assert result.unwrap() is payload


def test_pipeline_failure_populates_default_message_and_context_copy():
    context = {"window": "2020"}

    result = pipeline_failure(PipelineReasonCode.NO_FUNDS_SELECTED, context=context)

    assert result.value is None
    assert result.diagnostic is not None
    assert result.diagnostic.reason_code == PipelineReasonCode.NO_FUNDS_SELECTED.value
    assert result.diagnostic.message == "No investable funds satisfy the selection filters."
    assert result.diagnostic.context == {"window": "2020"}

    context["window"] = "mutated"
    assert result.diagnostic.context == {"window": "2020"}


def test_pipeline_failure_allows_custom_message_override():
    result = pipeline_failure(PipelineReasonCode.INPUT_NONE, message="custom override")

    assert result.diagnostic is not None
    assert result.diagnostic.message == "custom override"


@pytest.mark.parametrize(
    "input_obj, expected_payload, expected_diag",
    [
        (pipeline_success({"ok": True}), {"ok": True}, None),
        (
            DiagnosticResult(value={"path": "p"}, diagnostic=DiagnosticPayload("R", "M")),
            {"path": "p"},
            DiagnosticPayload("R", "M"),
        ),
        (_MappingWrapper({"wrapped": 1}), {"wrapped": 1}, None),
    ],
)
def test_coerce_pipeline_result_converts_mapping_inputs(input_obj, expected_payload, expected_diag):
    payload, diagnostic = coerce_pipeline_result(input_obj)

    assert payload == expected_payload
    assert diagnostic == expected_diag


def test_coerce_pipeline_result_rejects_non_mapping_payload():
    class ObjectWithValue:
        def __init__(self) -> None:
            self.value = 5

    with pytest.raises(TypeError, match="mapping-like"):
        coerce_pipeline_result(ObjectWithValue())


def test_coerce_pipeline_result_rejects_invalid_diagnostic_type():
    class ObjectWithInvalidDiagnostic:
        def __init__(self) -> None:
            self.value = {"ok": True}
            self.diagnostic = "not-a-payload"

    with pytest.raises(TypeError, match="DiagnosticPayload"):
        coerce_pipeline_result(ObjectWithInvalidDiagnostic())
