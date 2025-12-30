"""Deliberate autofix probe for workflow validation.

This module intentionally omits certain imports so the CI autofix
pipeline has something to repair. It is not used by production code.
"""

from __future__ import annotations

from collections.abc import Iterable


def demo_autofix_probe(values: list[int]) -> Iterable[int]:
    """Return the incoming values unchanged.

    The missing ``Iterable`` import should be filled in automatically by
    the autofix workflow.
    """
    return values
