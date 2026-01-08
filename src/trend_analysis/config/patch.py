"""Structured config patch models for NL-driven updates."""

from __future__ import annotations

import difflib
import re
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .validation import ValidationResult, validate_config

_DOTPATH_RE = re.compile(r"^[A-Za-z0-9_-]+(\[\d+\])*(\.[A-Za-z0-9_-]+(\[\d+\])*)*$")
_JSON_POINTER_RE = re.compile(r"^(/[^/\s]+)+$")

_VOL_TARGET_RISK_THRESHOLD = 0.15


class RiskFlag(str, Enum):
    REMOVES_CONSTRAINT = "REMOVES_CONSTRAINT"
    INCREASES_LEVERAGE = "INCREASES_LEVERAGE"
    REMOVES_VALIDATION = "REMOVES_VALIDATION"
    BROAD_SCOPE = "BROAD_SCOPE"


class PatchOperation(BaseModel):
    op: Literal["set", "remove", "append", "merge"] = Field(
        description="Operation type to apply at the path."
    )
    path: str = Field(description="JSONPointer or dotpath targeting the config.")
    value: Any | None = Field(
        default=None,
        description="Value for set/append/merge operations.",
    )
    rationale: str | None = Field(default=None, description="Optional explanation for the change.")

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
            if self.op == "append" and self.value is None:
                raise ValueError("value must be non-null for op 'append'")
            if self.op == "merge" and not isinstance(self.value, dict):
                raise ValueError("value must be an object for op 'merge'")
        elif self.op == "remove" and self.value is not None:
            raise ValueError("value must be null for op 'remove'")
        return self


class ConfigPatch(BaseModel):
    operations: list[PatchOperation] = Field(
        description="Ordered list of operations for the patch."
    )
    needs_review: bool = Field(
        default=False,
        description="True when the patch references unknown or ambiguous keys.",
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list,
        description="Detected or caller-provided risk flags.",
    )
    summary: str = Field(description="Human-readable summary of changes.")

    model_config = ConfigDict(extra="forbid")

    @field_validator("summary")
    @classmethod
    def _summary_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must be a non-empty string")
        return value

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


def apply_patch(config: dict[str, Any], patch: ConfigPatch) -> dict[str, Any]:
    """Apply a validated patch to a config mapping."""

    updated = deepcopy(config)
    for operation in patch.operations:
        try:
            segments = _parse_path_segments(operation.path)
            if not segments:
                raise ValueError("path must include at least one segment")
            parent, leaf = _resolve_parent(
                updated, segments, operation.op, allow_missing=operation.op == "remove"
            )
            if parent is None:
                continue
            # Type narrowing: if parent is not None, leaf is also not None
            assert leaf is not None
            _apply_operation(parent, leaf, operation)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise _path_error(operation.path, updated, exc) from exc
    return updated


def apply_config_patch(config: dict[str, Any], patch: ConfigPatch) -> dict[str, Any]:
    """Apply a validated patch to a config mapping."""

    return apply_patch(config, patch)


def diff_configs(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Return a unified diff for the YAML representation of two configs.

    Note: PyYAML does not preserve comments; callers using ruamel.yaml can
    supply comment-preserving YAML in their own wrappers if needed.
    """

    old_yaml = yaml.safe_dump(old, sort_keys=False, default_flow_style=False)
    new_yaml = yaml.safe_dump(new, sort_keys=False, default_flow_style=False)
    diff = difflib.unified_diff(
        old_yaml.splitlines(keepends=True),
        new_yaml.splitlines(keepends=True),
        fromfile="before",
        tofile="after",
        n=3,
    )
    return "".join(diff)


def apply_and_diff(config_path: Path, patch: ConfigPatch) -> tuple[dict[str, Any], str]:
    """Load a YAML config, apply a patch, and return the updated config + diff."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = payload or {}
    updated = apply_patch(config, patch)
    diff = diff_configs(config, updated)
    return updated, diff


def apply_and_validate(
    config: dict[str, Any], patch: ConfigPatch
) -> tuple[dict[str, Any], ValidationResult]:
    """Apply a patch and validate the updated configuration."""

    updated = apply_patch(config, patch)
    result = validate_config(updated)
    return updated, result


def _to_dotpath(path: str) -> str:
    segments = _parse_path_segments(path)
    return _format_dotpath(segments)


def _parse_path_segments(path: str) -> list[str | int]:
    if path.startswith("/"):
        segments = [
            segment.replace("~1", "/").replace("~0", "~") for segment in path.split("/")[1:]
        ]
        return [int(segment) if segment.isdigit() else segment for segment in segments]
    return _parse_dotpath(path) if path else []


def _parse_dotpath(path: str) -> list[str | int]:
    segments: list[str | int] = []
    for part in path.split("."):
        match = re.fullmatch(r"([A-Za-z0-9_-]+)((?:\[\d+\])*)", part)
        if not match:
            raise ValueError("path must be a dotpath or JSONPointer")
        key, indexes = match.groups()
        segments.append(key)
        if indexes:
            for idx in re.findall(r"\[(\d+)\]", indexes):
                segments.append(int(idx))
    return segments


def _format_dotpath(segments: list[str | int]) -> str:
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, int):
            if not parts:
                parts.append(f"[{segment}]")
            else:
                parts[-1] += f"[{segment}]"
        else:
            parts.append(segment)
    return ".".join(parts)


def _resolve_parent(
    root: dict[str, Any],
    segments: list[str | int],
    op: str,
    *,
    allow_missing: bool = False,
) -> tuple[Any, str | int] | tuple[None, None]:
    if not segments:
        raise ValueError("path must include at least one segment")
    current: Any = root
    for index, segment in enumerate(segments[:-1]):
        next_segment = segments[index + 1]
        if isinstance(segment, str):
            if not isinstance(current, dict):
                raise TypeError(f"path segment '{segment}' requires an object container")
            if segment not in current or current[segment] is None:
                if allow_missing:
                    return None, None
                current[segment] = [] if isinstance(next_segment, int) else {}
            current = current[segment]
        else:
            if not isinstance(current, list):
                raise TypeError(f"path segment '[{segment}]' requires a list container")
            if segment < 0:
                raise IndexError("list indices must be non-negative")
            if segment >= len(current):
                if allow_missing:
                    return None, None
                if op in {"set", "append", "merge"}:
                    current.extend([None] * (segment - len(current) + 1))
                else:
                    raise KeyError(f"path segment '[{segment}]' does not exist")
            if current[segment] is None:
                if allow_missing:
                    return None, None
                current[segment] = [] if isinstance(next_segment, int) else {}
            current = current[segment]
    return current, segments[-1]


def _apply_operation(parent: Any, leaf: str | int, operation: PatchOperation) -> None:
    if operation.op == "remove":
        if isinstance(leaf, int):
            if not isinstance(parent, list):
                raise TypeError("remove requires a list at the target path")
            if 0 <= leaf < len(parent):
                parent.pop(leaf)
            return
        if not isinstance(parent, dict):
            raise TypeError("remove requires an object container at the target path")
        parent.pop(leaf, None)
        return

    if operation.op == "set":
        if isinstance(leaf, int):
            if not isinstance(parent, list):
                raise TypeError("set requires a list at the target path")
            if leaf < 0:
                raise IndexError("list indices must be non-negative")
            if leaf >= len(parent):
                parent.extend([None] * (leaf - len(parent) + 1))
            parent[leaf] = operation.value
            return
        if not isinstance(parent, dict):
            raise TypeError("set requires an object container at the target path")
        # Check if key is a likely typo of an existing key
        if leaf not in parent and parent:
            close = difflib.get_close_matches(leaf, parent.keys(), n=1, cutoff=0.6)
            if close:
                raise KeyError(f"path segment '{leaf}' does not exist")
        parent[leaf] = operation.value
        return

    if operation.op == "append":
        if isinstance(leaf, int):
            raise TypeError("append requires a list at the target path, not an index")
        if not isinstance(parent, dict):
            raise TypeError("append requires an object container at the target path")
        existing = parent.get(leaf)
        if existing is None:
            parent[leaf] = [operation.value]
        elif isinstance(existing, list):
            existing.append(operation.value)
        else:
            raise TypeError("append requires a list at the target path")
        return

    if operation.op == "merge":
        if isinstance(leaf, int):
            if not isinstance(parent, list):
                raise TypeError("merge requires a list at the target path")
            if leaf < 0:
                raise IndexError("list indices must be non-negative")
            if leaf >= len(parent):
                parent.extend([None] * (leaf - len(parent) + 1))
            if parent[leaf] is None:
                parent[leaf] = {}
            target = parent[leaf]
        else:
            if not isinstance(parent, dict):
                raise TypeError("merge requires an object container at the target path")
            if leaf not in parent or parent[leaf] is None:
                parent[leaf] = {}
            target = parent[leaf]
        if not isinstance(target, dict):
            raise TypeError("merge requires an object at the target path")
        if not isinstance(operation.value, dict):
            raise TypeError("merge operation requires a dict value")
        _deep_merge(target, operation.value)
        return

    raise ValueError(f"unsupported operation '{operation.op}'")


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


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
    return len(_parse_path_segments(path))


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
        dotpath == prefix or dotpath.startswith(f"{prefix}.") for prefix in constraint_prefixes
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
        return isinstance(op.value, (int, float)) and op.value > _VOL_TARGET_RISK_THRESHOLD
    if dotpath == "vol_adjust" and isinstance(op.value, dict):
        target_vol = op.value.get("target_vol")
        return isinstance(target_vol, (int, float)) and target_vol > _VOL_TARGET_RISK_THRESHOLD
    return False


def _is_broad_scope(op: PatchOperation) -> bool:
    if op.op == "append":
        return False
    return _path_depth(op.path) <= 1


def _collect_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)
            paths.extend(_collect_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.append(path)
            paths.extend(_collect_paths(child, path))
    return paths


def _path_error(path: str, config: dict[str, Any], exc: Exception) -> ValueError:
    dotpath = _to_dotpath(path)
    candidates = _collect_paths(config)
    suggestions = difflib.get_close_matches(dotpath, candidates, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    return ValueError(f"Invalid path '{path}': {exc}.{hint}")


__all__ = [
    "apply_patch",
    "apply_config_patch",
    "ConfigPatch",
    "PatchOperation",
    "RiskFlag",
    "apply_and_diff",
    "apply_and_validate",
    "diff_configs",
]
