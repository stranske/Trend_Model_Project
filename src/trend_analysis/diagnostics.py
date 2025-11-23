"""Diagnostics helpers for the trend-analysis pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from trend.diagnostics import DiagnosticPayload, DiagnosticResult

AnalysisResult = dict[str, object]
PipelineResult = DiagnosticResult[AnalysisResult]


class PipelineReasonCode(str, Enum):
    """Canonical reason codes for pipeline early exits."""

    INPUT_NONE = "PIPELINE_INPUT_NONE"
    NO_VALID_DATES = "PIPELINE_NO_VALID_DATES"
    CALENDAR_ALIGNMENT_WIPE = "PIPELINE_CALENDAR_ALIGNMENT_WIPE"
    PREPARED_FRAME_EMPTY = "PIPELINE_PREPARED_FRAME_EMPTY"
    NO_VALUE_COLUMNS = "PIPELINE_NO_VALUE_COLUMNS"
    INSUFFICIENT_COLUMNS = "PIPELINE_INSUFFICIENT_COLUMNS"
    SAMPLE_WINDOW_EMPTY = "PIPELINE_SAMPLE_WINDOW_EMPTY"
    NO_FUNDS_SELECTED = "PIPELINE_NO_FUNDS_SELECTED"


_DEFAULT_MESSAGES: Mapping[PipelineReasonCode, str] = {
    PipelineReasonCode.INPUT_NONE: "Input DataFrame must be provided.",
    PipelineReasonCode.NO_VALID_DATES: "No valid timestamps were found in the Date column.",
    PipelineReasonCode.CALENDAR_ALIGNMENT_WIPE: "Calendar alignment removed every observation.",
    PipelineReasonCode.PREPARED_FRAME_EMPTY: "All rows were dropped during preprocessing.",
    PipelineReasonCode.NO_VALUE_COLUMNS: "No fund or index columns remain after preprocessing.",
    PipelineReasonCode.INSUFFICIENT_COLUMNS: "Insufficient data columns to continue the analysis.",
    PipelineReasonCode.SAMPLE_WINDOW_EMPTY: "In-sample or out-of-sample window is empty.",
    PipelineReasonCode.NO_FUNDS_SELECTED: "No investable funds satisfy the selection filters.",
}


def pipeline_success(value: AnalysisResult) -> PipelineResult:
    """Return a successful pipeline diagnostic wrapper."""

    return DiagnosticResult.success(value)


def pipeline_failure(
    code: PipelineReasonCode,
    *,
    message: str | None = None,
    context: Mapping[str, object] | None = None,
) -> PipelineResult:
    """Create a failure diagnostic populated with pipeline metadata."""

    default_message = _DEFAULT_MESSAGES.get(code)
    payload_message: str = message or default_message or code.value
    payload = DiagnosticPayload(
        reason_code=code.value,
        message=payload_message,
        context=dict(context) if context else None,
    )
    return DiagnosticResult(value=None, diagnostic=payload)


__all__ = [
    "AnalysisResult",
    "PipelineResult",
    "PipelineReasonCode",
    "pipeline_failure",
    "pipeline_success",
]
