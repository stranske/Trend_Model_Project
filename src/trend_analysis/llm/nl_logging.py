"""Structured logging models for NL operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from trend_analysis.config.patch import ConfigPatch
from trend_analysis.config.validation import ValidationResult


class NLOperationLog(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    timestamp: datetime
    operation: Literal["nl_to_patch", "apply_patch", "validate", "run"]
    input_hash: str
    prompt_template: str
    prompt_variables: dict[str, Any]
    model_output: str | None
    parsed_patch: ConfigPatch | None
    validation_result: ValidationResult | None
    error: str | None
    duration_ms: float
    model_name: str
    temperature: float
    token_usage: dict[str, Any] | None


__all__ = ["NLOperationLog"]
