from collections.abc import Mapping

import pytest

from trend.diagnostics import DiagnosticPayload, DiagnosticResult
from trend_analysis import diagnostics


def test_pipeline_success_exposes_mapping_interface():
    payload = {"alpha": 1, "beta": 2}
    result = diagnostics.pipeline_success(payload)

    assert result.unwrap() is payload
    assert dict(result) == payload
    assert list(result.keys()) == list(payload.keys())
    assert list(result.items()) == list(payload.items())
    assert list(result.values()) == list(payload.values())
    assert result.get("missing", "fallback") == "fallback"
    assert result.diagnostic is None


def test_pipeline_result_requires_value_for_item_access():
    result = diagnostics.PipelineResult(value=None)

    with pytest.raises(KeyError, match="does not contain a payload"):
        _ = result["anything"]


def test_pipeline_failure_defaults_message_and_copies_context():
    context = {"window": "empty"}
    result = diagnostics.pipeline_failure(
        diagnostics.PipelineReasonCode.SAMPLE_WINDOW_EMPTY, context=context
    )

    assert result.unwrap() is None
    diagnostic = result.diagnostic
    assert isinstance(diagnostic, DiagnosticPayload)
    assert (
        diagnostic.message
        == diagnostics._DEFAULT_MESSAGES[
            diagnostics.PipelineReasonCode.SAMPLE_WINDOW_EMPTY
        ]
    )
    assert diagnostic.context == context
    assert diagnostic.context is not context


def test_coerce_pipeline_result_accepts_diagnostic_result():
    diag_result = DiagnosticResult(value={"score": 5})
    payload, diagnostic = diagnostics.coerce_pipeline_result(diag_result)

    assert payload == {"score": 5}
    assert diagnostic is None


def test_coerce_pipeline_result_rejects_invalid_diagnostic_type():
    class PayloadHolder:
        diagnostic = object()
        value = {"x": 1}

    with pytest.raises(TypeError, match="DiagnosticPayload"):
        diagnostics.coerce_pipeline_result(PayloadHolder())


def test_coerce_pipeline_result_rejects_non_mapping_payload():
    diag_result = DiagnosticResult(value=[1, 2, 3])

    with pytest.raises(TypeError, match="mapping-like"):
        diagnostics.coerce_pipeline_result(diag_result)


def test_pipeline_failure_accepts_custom_message_and_context_copy(monkeypatch):
    custom_message = "custom override"
    context = {"reason": "manual"}
    result = diagnostics.pipeline_failure(
        diagnostics.PipelineReasonCode.NO_FUNDS_SELECTED,
        message=custom_message,
        context=context,
    )

    diagnostic = result.diagnostic
    assert diagnostic is not None
    assert diagnostic.message == custom_message
    assert diagnostic.context == context
    assert diagnostic.context is not context


def test_pipeline_failure_falls_back_to_reason_code_when_default_missing(monkeypatch):
    monkeypatch.delitem(
        diagnostics._DEFAULT_MESSAGES,
        diagnostics.PipelineReasonCode.INDICES_ABSENT,
        raising=False,
    )

    result = diagnostics.pipeline_failure(
        diagnostics.PipelineReasonCode.INDICES_ABSENT,
    )

    diagnostic = result.diagnostic
    assert diagnostic is not None
    assert diagnostic.message == diagnostics.PipelineReasonCode.INDICES_ABSENT.value


def test_coerce_pipeline_result_converts_mapping_payload_and_preserves_diagnostic():
    class MappingPayload(dict):
        pass

    payload = MappingPayload({"alpha": 1})
    diagnostic = DiagnosticPayload(
        diagnostics.PipelineReasonCode.NO_VALUE_COLUMNS.value, "missing values"
    )

    class Wrapper:
        def __init__(self) -> None:
            self.value = payload
            self.diagnostic = diagnostic

    coerced_payload, coerced_diag = diagnostics.coerce_pipeline_result(Wrapper())

    assert coerced_payload == {"alpha": 1}
    assert isinstance(coerced_payload, dict)
    assert coerced_diag is diagnostic


def test_pipeline_result_len_iter_and_copy_with_none_value():
    result = diagnostics.PipelineResult(value=None)

    assert len(result) == 0
    assert list(result) == []
    assert result.copy() == {}
    assert result.get("missing", "default") == "default"


def test_coerce_pipeline_result_wraps_custom_mapping_into_dict():
    class CustomMapping(Mapping):
        def __init__(self) -> None:
            self._data = {"alpha": 1, "beta": 2}

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    payload = CustomMapping()

    coerced_payload, diagnostic = diagnostics.coerce_pipeline_result(payload)

    assert diagnostic is None
    assert coerced_payload == {"alpha": 1, "beta": 2}
    assert isinstance(coerced_payload, dict)
