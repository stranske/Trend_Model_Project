"""Diagnostics helpers for the trend-analysis pipeline."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, cast

from trend.diagnostics import DiagnosticPayload, DiagnosticResult

AnalysisResult = dict[str, object]


@dataclass(slots=True)
class PipelineResult(Mapping[str, object]):
    """Dictionary-like container pairing a payload with diagnostics."""

    value: AnalysisResult | None
    diagnostic: DiagnosticPayload | None = None

    def _require_value(self) -> AnalysisResult:
        if self.value is None:
            raise KeyError("Pipeline result does not contain a payload")
        return self.value

    def __getitem__(self, key: str) -> object:
        return self._require_value()[key]

    def __iter__(self) -> Iterator[str]:
        data = self.value or {}
        return iter(data)

    def __len__(self) -> int:
        data = self.value or {}
        return len(data)

    def __bool__(self) -> bool:  # pragma: no cover - trivial truthiness
        return bool(self.value)

    def get(self, key: str, default: object | None = None) -> object | None:
        if self.value is None:
            return default
        return self.value.get(key, default)

    def keys(self) -> KeysView[str]:
        data = self.value or {}
        return data.keys()

    def items(self) -> ItemsView[str, object]:
        data = self.value or {}
        return data.items()

    def values(self) -> ValuesView[object]:
        data = self.value or {}
        return data.values()

    def copy(self) -> AnalysisResult:
        data = self.value or {}
        return dict(data)

    def unwrap(self) -> AnalysisResult | None:
        """Return the underlying analysis payload without copying."""

        return self.value


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
    INDICES_ABSENT = "PIPELINE_INDICES_ABSENT"


_DEFAULT_MESSAGES: Mapping[PipelineReasonCode, str] = {
    PipelineReasonCode.INPUT_NONE: "Input DataFrame must be provided.",
    PipelineReasonCode.NO_VALID_DATES: "No valid timestamps were found in the Date column.",
    PipelineReasonCode.CALENDAR_ALIGNMENT_WIPE: "Calendar alignment removed every observation.",
    PipelineReasonCode.PREPARED_FRAME_EMPTY: "All rows were dropped during preprocessing.",
    PipelineReasonCode.NO_VALUE_COLUMNS: "No fund or index columns remain after preprocessing.",
    PipelineReasonCode.INSUFFICIENT_COLUMNS: "Insufficient data columns to continue the analysis.",
    PipelineReasonCode.SAMPLE_WINDOW_EMPTY: "In-sample or out-of-sample window is empty.",
    PipelineReasonCode.NO_FUNDS_SELECTED: "No investable funds satisfy the selection filters.",
    PipelineReasonCode.INDICES_ABSENT: "Requested indices are missing from the analysis window.",
}


def pipeline_success(value: AnalysisResult) -> PipelineResult:
    """Return a successful pipeline diagnostic wrapper."""

    return PipelineResult(value=value, diagnostic=None)


def pipeline_failure(
    code: PipelineReasonCode,
    *,
    message: str | None = None,
    context: Mapping[str, object] | None = None,
) -> PipelineResult:
    """Create a failure diagnostic populated with pipeline metadata."""

    default_message = _DEFAULT_MESSAGES.get(code)
    # Fallback order: (1) custom message, (2) default message, (3) enum value itself
    if message is not None:
        payload_message = message
    elif default_message is not None:
        payload_message = default_message
    else:
        payload_message = code.value
    payload = DiagnosticPayload(
        reason_code=code.value,
        message=payload_message,
        context=dict(context) if context else None,
    )
    return PipelineResult(value=None, diagnostic=payload)


def coerce_pipeline_result(
    result: object,
) -> Tuple[AnalysisResult | None, DiagnosticPayload | None]:
    """Return a ``(payload, diagnostic)`` pair for arbitrary pipeline outputs."""

    if isinstance(result, PipelineResult):
        return result.value, result.diagnostic

    diagnostic_attr = getattr(result, "diagnostic", None)
    if diagnostic_attr is not None and not isinstance(
        diagnostic_attr, DiagnosticPayload
    ):
        raise TypeError(
            "Pipeline diagnostics must be DiagnosticPayload instances; received "
            f"{type(diagnostic_attr)!r}"
        )
    diagnostic: DiagnosticPayload | None = diagnostic_attr

    if isinstance(result, DiagnosticResult):
        payload = cast(AnalysisResult | None, result.value)
    elif isinstance(result, Mapping):
        payload = cast(AnalysisResult | None, result)
    elif hasattr(result, "value"):
        payload = cast(AnalysisResult | None, getattr(result, "value"))
    else:
        payload = cast(AnalysisResult | None, result)

    if payload is None:
        return None, diagnostic

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Pipeline outputs must be mapping-like; received {type(payload)!r}"
        )

    if isinstance(payload, dict):
        return payload, diagnostic
    return dict(payload), diagnostic


__all__ = [
    "AnalysisResult",
    "PipelineResult",
    "PipelineReasonCode",
    "pipeline_failure",
    "pipeline_success",
    "coerce_pipeline_result",
]
