"""Tier 3: economic invariants on the baseline and on every scenario variant."""

from __future__ import annotations

import pytest
from baseline_kit import assert_invariants

from . import invariants
from .conftest import load_catalog
from .harness import run_scenario

_SCENARIOS = load_catalog()["scenarios"]
_SCEN_IDS = [s["id"] for s in _SCENARIOS]


def _effective(patch: dict, key: str, default):
    return patch.get(key, default)


def _assert_invariants(out, *, long_only, max_weight, context=""):
    results = invariants.check_all(out, long_only=long_only, max_weight=max_weight)
    assert_invariants(results, context=context)


def test_baseline_invariants(baseline_output):
    _assert_invariants(baseline_output, long_only=True, max_weight=0.25)


@pytest.mark.parametrize("scen", _SCENARIOS, ids=_SCEN_IDS)
def test_scenario_invariants(scen):
    patch = {**(scen.get("base") or {}), **(scen.get("vary") or {})}
    out = run_scenario("config/demo.yml", patch)
    long_only = bool(_effective(patch, "portfolio.constraints.long_only", True))
    max_weight = _effective(patch, "portfolio.constraints.max_weight", 0.25)
    _assert_invariants(out, long_only=long_only, max_weight=max_weight, context=scen["id"])
