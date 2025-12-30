"""Automation fixture used by diagnostic pipelines.

Historically this helper intentionally violated its return annotation so
the automation demos could showcase mypy repairs. The real workflows now
expect the implementation to be sound while still returning the pipe-
delimited string consumed by downstream tests.
"""

from __future__ import annotations

from collections.abc import Iterable


def aggregate_numbers(values: Iterable[int]) -> str:
    """Return the diagnostic pipe-separated string used in the demo suite."""
    return " | ".join(str(v) for v in values)
