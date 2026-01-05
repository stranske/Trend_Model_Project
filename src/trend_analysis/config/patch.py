"""Structured config patch models for NL-driven updates."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DOTPATH_RE = re.compile(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$")
_JSON_POINTER_RE = re.compile(r"^(/[^/\s]+)+$")

_VOL_TARGET_RISK_THRESHOLD = 0.15


class RiskFlag(str, Enum):
    REMOVES_CONSTRAINT = "REMOVES_CONSTRAINT"
    INCREASES_LEVERAGE = "INCREASES_LEVERAGE"
    REMOVES_VALIDATION = "REMOVES_VALIDATION"
    BROAD_SCOPE = "BROAD_SCOPE"


class PatchOperation(BaseModel):
    op: Literal["set", "remove", "append", "merge"]
    path: str
    value: Any | None = None
    rationale: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_operation(self) -> "PatchOperation":
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("path must be a non-empty string")
        if self.path.startswith("/"):
            if self.path == "/":
                raise ValueError("path must include at least one segment")
            if not _JSON_POINTER_RE.match(self.path):
                raise ValueError("path must be a valid JSONPointer")
            if _has_invalid_json_pointer_escape(self.path):
                raise ValueError("path must use valid JSONPointer escape sequences")
        else:
            if not _DOTPATH_RE.match(self.path):
                raise ValueError("path must be a dotpath or JSONPointer")
        if self.op in {"set", "append", "merge"}:
            if "value" not in self.model_fields_set:
                raise ValueError(f"value is required for op '{self.op}'")
            if self.op == "merge" and not isinstance(self.value, dict):
                raise ValueError("value must be an object for op 'merge'")
        elif self.op == "remove" and self.value is not None:
            raise ValueError("value must be null for op 'remove'")
        return self


class ConfigPatch(BaseModel):
    operations: list[PatchOperation]
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    summary: str

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _populate_risk_flags(self) -> "ConfigPatch":
        detected = _detect_risk_flags(self.operations)
        if self.risk_flags:
            combined = {flag for flag in self.risk_flags}
            combined.update(detected)
            self.risk_flags = [flag for flag in RiskFlag if flag in combined]
        else:
            self.risk_flags = detected
        return self

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return JSON schema for LLM prompting."""

        return cls.model_json_schema()


def _to_dotpath(path: str) -> str:
    if path.startswith("/"):
        segments = [
            segment.replace("~1", "/").replace("~0", "~")
            for segment in path.split("/")[1:]
        ]
        return ".".join(segments)
    return path


def _has_invalid_json_pointer_escape(path: str) -> bool:
    for segment in path.split("/")[1:]:
        index = 0
        while index < len(segment):
            if segment[index] != "~":
                index += 1
                continue
            if index + 1 >= len(segment) or segment[index + 1] not in {"0", "1"}:
                return True
            index += 2
    return False


def _path_depth(path: str) -> int:
    if path.startswith("/"):
        return len([segment for segment in path.split("/")[1:] if segment])
    return len(path.split("."))


def _detect_risk_flags(operations: list[PatchOperation]) -> list[RiskFlag]:
    flags: set[RiskFlag] = set()
    for op in operations:
        dotpath = _to_dotpath(op.path)
        if _removes_constraint(op, dotpath):
            flags.add(RiskFlag.REMOVES_CONSTRAINT)
        if _removes_validation(op, dotpath):
            flags.add(RiskFlag.REMOVES_VALIDATION)
        if _increases_leverage(op, dotpath):
            flags.add(RiskFlag.INCREASES_LEVERAGE)
        if _is_broad_scope(op):
            flags.add(RiskFlag.BROAD_SCOPE)
    return [flag for flag in RiskFlag if flag in flags]


def _removes_constraint(op: PatchOperation, dotpath: str) -> bool:
    constraint_prefixes = (
        "portfolio.constraints",
        "portfolio.max_turnover",
        "portfolio.leverage_cap",
    )
    removes_value = op.op == "remove" or (op.op == "set" and op.value is None)
    return removes_value and any(
        dotpath == prefix or dotpath.startswith(f"{prefix}.")
        for prefix in constraint_prefixes
    )


def _removes_validation(op: PatchOperation, dotpath: str) -> bool:
    validation_paths = (
        "robustness",
        "robustness.condition_check",
        "robustness.condition_check.enabled",
        "robustness.shrinkage.enabled",
    )
    removes_value = op.op == "remove" or (op.op == "set" and op.value is None)
    disables = op.op == "set" and op.value is False
    return (removes_value or disables) and any(
        dotpath == path or dotpath.startswith(f"{path}.") for path in validation_paths
    )


def _increases_leverage(op: PatchOperation, dotpath: str) -> bool:
    if op.op not in {"set", "merge"}:
        return False
    if dotpath == "vol_adjust.target_vol":
        return (
            isinstance(op.value, (int, float)) and op.value > _VOL_TARGET_RISK_THRESHOLD
        )
    if dotpath == "vol_adjust" and isinstance(op.value, dict):
        target_vol = op.value.get("target_vol")
        return (
            isinstance(target_vol, (int, float))
            and target_vol > _VOL_TARGET_RISK_THRESHOLD
        )
    return False


def _is_broad_scope(op: PatchOperation) -> bool:
    if op.op == "append":
        return False
    return _path_depth(op.path) <= 1


__all__ = [
    "ConfigPatch",
    "PatchOperation",
    "RiskFlag",
]
