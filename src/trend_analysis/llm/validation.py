"""Validation helpers for ConfigPatch operations against the config schema."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import re
from typing import Any, Iterable

from trend_analysis.config.patch import PatchOperation

_DOTPATH_RE = re.compile(r"^[A-Za-z0-9_-]+(\[\d+\])*(\.[A-Za-z0-9_-]+(\[\d+\])*)*$")


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
        if not _path_exists(schema, segments):
            dotpath = _format_dotpath(segments)
            suggestion = _suggest_path(dotpath, candidates)
            unknown.append(UnknownKey(path=dotpath, suggestion=suggestion))
    return unknown


def _parse_path_segments(path: str) -> list[str | int]:
    if path.startswith("/"):
        segments = [
            segment.replace("~1", "/").replace("~0", "~") for segment in path.split("/")[1:]
        ]
        return [int(segment) if segment.isdigit() else segment for segment in segments]
    if not _DOTPATH_RE.match(path):
        return [path]
    segments: list[str | int] = []
    for part in path.split("."):
        match = re.fullmatch(r"([A-Za-z0-9_-]+)((?:\[\d+\])*)", part)
        if not match:
            segments.append(part)
            continue
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


def _path_exists(schema: dict[str, Any], segments: list[str | int]) -> bool:
    current: Any = schema
    for segment in segments:
        if isinstance(segment, int):
            if not isinstance(current, dict):
                return False
            items = current.get("items")
            if not isinstance(items, dict):
                return False
            current = items
            continue
        if not isinstance(current, dict):
            return False
        properties = current.get("properties")
        if not isinstance(properties, dict) or segment not in properties:
            return False
        current = properties[segment]
    return True


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
    suggestions = difflib.get_close_matches(path, candidates, n=1, cutoff=0.6)
    return suggestions[0] if suggestions else None


__all__ = ["UnknownKey", "validate_patch_keys"]
