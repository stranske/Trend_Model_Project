"""Helpers to validate configuration files against the generated schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from utils.paths import proj_path

_DEFAULT_SCHEMA_PATH = proj_path() / "config.schema.json"


def load_schema(schema_path: Path = _DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Load the JSON schema from disk."""

    result: dict[str, Any] = json.loads(schema_path.read_text(encoding="utf-8"))
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file '{config_path}' must contain a mapping at the root.")
    return payload


def _format_error(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    location = path or "<root>"
    return f"{location}: {error.message}"


def validate_config_data(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate config payload and return a list of error strings."""

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda err: list(err.absolute_path))
    return [_format_error(error) for error in errors]


def validate_config_file(config_path: Path, schema_path: Path = _DEFAULT_SCHEMA_PATH) -> list[str]:
    """Validate a config file and return a list of errors."""

    schema = load_schema(schema_path)
    payload = load_config(config_path)
    return validate_config_data(payload, schema)


__all__ = ["validate_config_file"]
