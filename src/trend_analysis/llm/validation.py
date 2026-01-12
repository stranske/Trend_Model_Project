"""Validation helpers for ConfigPatch operations against the config schema."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from trend_analysis.config.patch import ConfigPatch, PatchOperation


class _ArrayWildcardMarker:
    pass


_ARRAY_WILDCARD = _ArrayWildcardMarker()
_Segment = str | int | _ArrayWildcardMarker

_DOTPATH_RE = re.compile(
    r"^(?:[A-Za-z0-9_-]+|\*)(?:\[(?:\d+|\*)\])*" r"(?:\.(?:[A-Za-z0-9_-]+|\*)(?:\[(?:\d+|\*)\])*)*$"
)


@dataclass(frozen=True)
class UnknownKey:
    """Details about a patch path that does not exist in the schema."""

    path: str
    suggestion: str | None = None


def validate_patch_keys(
    operations: Iterable[PatchOperation],
    schema: dict[str, Any] | None,
) -> list[UnknownKey]:
    """Return unknown keys referenced by patch operations."""

    if not schema:
        return []

    candidates = _collect_schema_paths(schema)
    unknown: list[UnknownKey] = []
    for operation in operations:
        segments = _parse_path_segments(operation.path)
        dotpath = _format_dotpath(segments)
        if not _path_exists(schema, segments):
            suggestion = _suggest_path(dotpath, candidates)
            unknown.append(UnknownKey(path=dotpath, suggestion=suggestion))
            continue
        if _has_dynamic_segments(segments):
            unknown.append(UnknownKey(path=dotpath))
    return unknown


def flag_unknown_keys(
    patch: ConfigPatch,
    schema: dict[str, Any] | None,
    *,
    logger: logging.Logger | None = None,
) -> list[UnknownKey]:
    """Mark patches with unknown keys as needing review and log details."""

    unknown = validate_patch_keys(patch.operations, schema)
    if not unknown:
        return []
    patch.needs_review = True
    active_logger = logger or logging.getLogger(__name__)
    for entry in unknown:
        if entry.suggestion:
            active_logger.warning(
                "Unknown config key '%s'. Did you mean '%s'?",
                entry.path,
                entry.suggestion,
            )
        else:
            active_logger.warning("Unknown config key '%s'.", entry.path)
    return unknown


def _parse_path_segments(path: str) -> list[_Segment]:
    if path.startswith("/"):
        json_segments = [
            segment.replace("~1", "/").replace("~0", "~") for segment in path.split("/")[1:]
        ]
        return [int(segment) if segment.isdigit() else segment for segment in json_segments]
    if not _DOTPATH_RE.match(path):
        return [path]
    segments: list[str | int] = []
    for part in path.split("."):
        match = re.fullmatch(r"([A-Za-z0-9_-]+|\*)((?:\[(?:\d+|\*)\])*)", part)
        if not match:
            segments.append(part)
            continue
        key, indexes = match.groups()
        segments.append(key)
        if indexes:
            for idx in re.findall(r"\[(\d+|\*)\]", indexes):
                segments.append(int(idx) if idx.isdigit() else _ARRAY_WILDCARD)
    return segments


def _format_dotpath(segments: list[_Segment]) -> str:
    parts: list[str] = []
    for segment in segments:
        if isinstance(segment, int) or segment is _ARRAY_WILDCARD:
            if not parts:
                parts.append("[*]" if segment is _ARRAY_WILDCARD else f"[{segment}]")
            else:
                parts[-1] += "[*]" if segment is _ARRAY_WILDCARD else f"[{segment}]"
        else:
            parts.append(segment)
    return ".".join(parts)


def _has_dynamic_segments(segments: list[_Segment]) -> bool:
    return any(
        isinstance(segment, int) or segment == "*" or segment is _ARRAY_WILDCARD
        for segment in segments
    )


def _is_array_segment(segment: _Segment) -> bool:
    return isinstance(segment, int) or segment is _ARRAY_WILDCARD


def _path_exists(schema: dict[str, Any], segments: list[_Segment]) -> bool:
    return _path_exists_at(schema, segments, 0)


def _path_exists_at(schema: dict[str, Any], segments: list[_Segment], index: int) -> bool:
    if index >= len(segments):
        return True
    current_segment = segments[index]
    if _is_array_segment(current_segment):
        return _path_exists_array(schema, segments, index)
    if current_segment == "*":
        return _path_exists_wildcard(schema, segments, index)
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    if isinstance(properties, dict) and current_segment in properties:
        return _path_exists_at(properties[current_segment], segments, index + 1)
    additional = schema.get("additionalProperties")
    if additional is True:
        return True
    if isinstance(additional, dict):
        return _path_exists_at(additional, segments, index + 1)
    return False


def _path_exists_array(schema: dict[str, Any], segments: list[_Segment], index: int) -> bool:
    if not isinstance(schema, dict):
        return False
    items = schema.get("items")
    if not isinstance(items, dict):
        return False
    return _path_exists_at(items, segments, index + 1)


def _path_exists_wildcard(schema: dict[str, Any], segments: list[_Segment], index: int) -> bool:
    if not isinstance(schema, dict):
        return False
    items = schema.get("items")
    if isinstance(items, dict):
        return _path_exists_at(items, segments, index + 1)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        return _path_exists_at(additional, segments, index + 1)
    if additional is True:
        return True
    properties = schema.get("properties")
    if isinstance(properties, dict):
        return any(_path_exists_at(value, segments, index + 1) for value in properties.values())
    return False


def _collect_schema_paths(schema: dict[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    if not isinstance(schema, dict):
        return paths
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key, value in properties.items():
            if not isinstance(key, str):
                continue
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)
            paths.extend(_collect_schema_paths(value, path))
    items = schema.get("items")
    if isinstance(items, dict) and prefix:
        paths.extend(_collect_schema_paths(items, prefix))
    return paths


def _suggest_path(path: str, candidates: list[str]) -> str | None:
    suggestions = difflib.get_close_matches(path, candidates, n=1, cutoff=0.8)
    return suggestions[0] if suggestions else None


__all__ = ["UnknownKey", "flag_unknown_keys", "validate_patch_keys"]
