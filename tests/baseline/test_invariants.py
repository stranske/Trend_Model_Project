"""Tier 3: economic invariants on the baseline and on every scenario variant."""

from __future__ import annotations

import pandas as pd
import pytest
from baseline_kit import assert_invariants

from . import invariants
from .conftest import load_catalog
from .harness import ScenarioOutput, run_scenario

_SCENARIOS = load_catalog()["scenarios"]
_SCEN_IDS = [s["id"] for s in _SCENARIOS]


def _effective(patch: dict, key: str, default):
    return patch.get(key, default)


def _assert_invariants(out, *, long_only, max_weight, context=""):
    results = invariants.check_all(out, long_only=long_only, max_weight=max_weight)
    assert_invariants(results, context=context)


def test_baseline_invariants(baseline_output):
    # The demo configures no max_weight cap (constraints: None), so don't assert one;
    # the configured rank-top-5 portfolio's natural max weight is not a cap violation.
    _assert_invariants(baseline_output, long_only=True, max_weight=None)


def _synthetic_output(weights: dict[str, float]) -> ScenarioOutput:
    """Minimal ScenarioOutput carrying a chosen set of fund weights."""
    w = pd.Series(weights, dtype=float)
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.01], dtype=float)
    return ScenarioOutput(
        metrics=pd.DataFrame(index=list(weights)),
        weights=w,
        fund_weights=w,
        turnover=pd.Series([0.0], dtype=float),
        portfolio=returns,
        costs={},
        seed=0,
    )


def _by_name(results):
    return {r.name: r for r in results}


def test_under_invested_book_fails_weight_sum_invariant():
    """An 80%-invested portfolio must fail `weight_sum_near_one`.

    Regression guard for #5915. These two conditions used to be OR-ed together,
    which made the weight-sum half unfalsifiable: gross exposure of 0.80 sits
    far below the 2.0 leverage cap, so the invariant passed. That is exactly how
    the demo baseline hid a benchmark holding 20% of the book (#5914).
    """
    # Four funds at 0.20 -- the precise shape the SPX bug produced.
    out = _synthetic_output({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2})
    results = _by_name(invariants.check_all(out, long_only=True, max_weight=None))

    assert results["weight_sum_near_one"].ok is False
    assert "0.80" in results["weight_sum_near_one"].detail

    # The leverage check must still pass -- proving the two are now independent
    # and that the old OR would have masked the failure above.
    assert results["gross_within_leverage_cap"].ok is True


def test_fully_invested_book_passes_both_weight_invariants():
    out = _synthetic_output({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})
    results = _by_name(invariants.check_all(out, long_only=True, max_weight=None))

    assert results["weight_sum_near_one"].ok is True
    assert results["gross_within_leverage_cap"].ok is True


def test_over_leveraged_book_fails_only_the_leverage_invariant():
    """A 3x long/short book breaches the cap while still summing to ~1."""
    out = _synthetic_output({"A": 2.0, "B": -1.0})
    results = _by_name(invariants.check_all(out, long_only=False, max_weight=None))

    assert results["weight_sum_near_one"].ok is True
    assert results["gross_within_leverage_cap"].ok is False


@pytest.mark.parametrize("scen", _SCENARIOS, ids=_SCEN_IDS)
def test_scenario_invariants(scen):
    patch = {**(scen.get("base") or {}), **(scen.get("vary") or {})}
    out = run_scenario("config/demo.yml", patch)
    long_only = bool(_effective(patch, "portfolio.constraints.long_only", True))
    # Only assert the max_weight cap when the scenario actually configures one.
    max_weight = _effective(patch, "portfolio.constraints.max_weight", None)
    _assert_invariants(out, long_only=long_only, max_weight=max_weight, context=scen["id"])
