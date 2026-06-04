"""Shared environment parsing helpers for LLM modules."""

from __future__ import annotations

import os


def read_env_float(name: str, *, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}.") from exc
