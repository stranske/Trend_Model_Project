"""Tier 1: directional ("metamorphic") sensibility checks.

Each scenario runs a control and a variant config and checks that the named
metric moves in the expected direction. Enforced scenarios fail on a wrong
direction; non-enforced ones record the observed direction (written to the
report) so we can confirm what "economically sensible" looks like before
promoting them to enforced.
"""

from __future__ import annotations

import pytest
from baseline_kit import evaluate_direction

from .conftest import load_catalog
from .harness import run_scenario

_TOL = 1e-9
_SCENARIOS = load_catalog()["scenarios"]
_SCEN_IDS = [s["id"] for s in _SCENARIOS]


def _metric(scen, patch):
    out = run_scenario("config/demo.yml", patch)
    if (
        scen.get("metric") == "num_selected"
        and scen.get("param") == "portfolio.rank.n"
        and out.declared_selected_count is not None
    ):
        return float(out.declared_selected_count)
    return float(out.derived()[scen["metric"]])


@pytest.mark.parametrize("scen", _SCENARIOS, ids=_SCEN_IDS)
def test_directional(scen, record_property):
    control_patch = {**(scen.get("base") or {}), **(scen.get("control") or {})}
    vary_patch = {**(scen.get("base") or {}), **(scen.get("vary") or {})}
    c = _metric(scen, control_patch)
    v = _metric(scen, vary_patch)

    direction = scen["direction"]
    observed = "↑" if v > c + _TOL else "↓" if v < c - _TOL else "≈"
    holds = evaluate_direction(direction, v, c, _TOL)

    msg = (
        f"{scen['id']}: {scen['metric']} control={c:.6g} variant={v:.6g} "
        f"observed={observed} expected={direction} holds={holds}"
    )
    record_property("directional", msg)

    if scen.get("enforce"):
        assert holds, "Economically wrong direction -- " + msg
    elif not holds:
        # Recorded for human confirmation; not a failure yet.
        pytest.skip("[report-only] " + msg)
