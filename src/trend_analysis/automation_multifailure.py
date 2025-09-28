"""Intentional type mismatch module for workflow automation validation.

This module deliberately returns the wrong type so that mypy detects an
error.
"""

from __future__ import annotations

from typing import Iterable


def aggregate_numbers(values: Iterable[int]) -> int:
    """Return a pipe-separated string while claiming to return an int."""
    return " | ".join(str(v) for v in values)
