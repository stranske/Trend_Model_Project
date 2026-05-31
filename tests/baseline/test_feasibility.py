"""Feasibility behavior for infeasible parameters (finding #4) -- PENDING DECISION.

The infeasibility guard was reverted: erroring on every ``max_weight * N < 1``
case breaks ~47 existing tests and blocks legitimately small portfolios (e.g. a
2-fund book at a 25% cap holding 50% cash). The correct behavior (error vs.
respect-cap-and-hold-cash, and over what scope) is an open owner decision.

These tests document the current behavior and will be turned into real
assertions once the policy is chosen.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "Finding #4: infeasible-max_weight policy not yet decided (see README).",
    allow_module_level=True,
)
