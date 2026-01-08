"""Helpers for schema selection in ConfigPatch prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, cast

from utils.paths import proj_path

_COMPACT_SCHEMA_PATH = proj_path("config.schema.compact.json")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_DOTPATH_RE = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+")


def load_compact_schema(path: Path | None = None) -> dict[str, Any]:
    """Load the compact config schema used for prompt injection."""

    target = path or _COMPACT_SCHEMA_PATH
    return cast(dict[str, Any], json.loads(target.read_text(encoding="utf-8")))


def select_schema_sections(schema: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Filter schema to sections likely relevant to the instruction."""

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema

    tokens = _extract_tokens(instruction)
    matched = {
        path[0]
        for path in _iter_schema_paths(properties)
        if _path_matches(path, tokens)
    }
    if not matched:
        return schema

    filtered: dict[str, Any] = {
        key: value
        for key, value in schema.items()
        if key not in {"properties", "default"}
    }
    filtered["properties"] = {
        key: properties[key] for key in matched if key in properties
    }

    defaults = schema.get("default")
    if isinstance(defaults, dict):
        filtered_defaults = {key: defaults[key] for key in matched if key in defaults}
        if filtered_defaults:
            filtered["default"] = filtered_defaults

    return filtered


def _extract_tokens(instruction: str) -> set[str]:
    text = instruction.lower()
    tokens = {match.lower() for match in _TOKEN_RE.findall(text)}
    for path in _DOTPATH_RE.findall(text):
        tokens.update(path.lower().split("."))
    return tokens


def _iter_schema_paths(properties: dict[str, Any]) -> Iterable[tuple[str, ...]]:
    for key, value in properties.items():
        if not isinstance(key, str):
            continue
        path = (key,)
        yield path
        nested = value.get("properties") if isinstance(value, dict) else None
        if isinstance(nested, dict):
            for child_path in _iter_schema_paths(nested):
                yield path + child_path


def _path_matches(path: tuple[str, ...], tokens: set[str]) -> bool:
    for segment in path:
        if _segment_matches(segment, tokens):
            return True
    return False


def _segment_matches(segment: str, tokens: set[str]) -> bool:
    lowered = segment.lower()
    if lowered in tokens:
        return True
    parts = [part for part in re.split(r"[_-]+", lowered) if part]
    return bool(parts) and all(part in tokens for part in parts)


__all__ = ["load_compact_schema", "select_schema_sections"]
