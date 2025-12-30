"""CI probe module (style-only) used to verify autofix pipeline.

This version intentionally keeps the code clean so the current iteration
can achieve a passing validation run; historical purpose (introducing
only auto-fixable issues) is documented in earlier commits.
"""

from __future__ import annotations

import math

import yaml

PROBE_VERSION = "style_only_v3"


def add_numbers(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def build_message(name: str = "World", excited: bool = False) -> str:
    """Return a greeting string."""
    msg = f"Hello {name}"
    return msg + ("!" if excited else "")


def _internal_helper(values: list[int]) -> int:
    """Return the sum of ``values`` after a trivial parse side-effect."""
    _ = yaml.safe_load("numbers: [1,2,3]")  # exercise import path
    _ = math.sqrt(values[0] if values else 0)
    return sum(values)


__all__ = ["add_numbers", "build_message", "_internal_helper", "PROBE_VERSION"]
