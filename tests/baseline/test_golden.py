"""Tier 0/1 golden masters via pytest-regressions.

Baselines are stored on disk (``tests/baseline/test_golden/*.npz`` etc.) and
diffed with float tolerance on every run. To (re-)bless after an *intended*
change:  ``pytest tests/baseline/test_golden.py --force-regen``
then review the diff and commit the updated baseline files.
"""

from __future__ import annotations

import pytest
from baseline_kit import DEFAULT_TOLERANCE, check_metrics

from .conftest import load_catalog
from .harness import run_scenario

TOL = DEFAULT_TOLERANCE
_SCENARIOS = load_catalog()["scenarios"]
_SCEN_IDS = [s["id"] for s in _SCENARIOS]


def test_baseline_derived_metrics(baseline_output, num_regression):
    """Economic summary stats of the reference run."""
    check_metrics(num_regression, baseline_output.derived())


def test_baseline_metric_columns(baseline_output, num_regression):
    """Every numeric column of the per-fund metrics table."""
    frame = baseline_output.metrics
    data = {str(col): frame[col].to_numpy(dtype=float) for col in frame.columns}
    num_regression.check(data, default_tolerance=TOL)


def test_baseline_fund_weights(baseline_output, num_regression):
    """Final portfolio weights (benchmarks excluded)."""
    w = baseline_output.fund_weights
    num_regression.check({"weight": w.to_numpy(dtype=float)}, default_tolerance=TOL)


@pytest.mark.parametrize("scen", _SCENARIOS, ids=_SCEN_IDS)
def test_scenario_variant_golden(scen, num_regression):
    """Golden-master the derived metrics of each Tier-1 *variant* config too,
    so future versions are pinned across the whole scenario grid (not just the
    baseline)."""
    patch = {**(scen.get("base") or {}), **(scen.get("vary") or {})}
    out = run_scenario("config/demo.yml", patch)
    check_metrics(num_regression, out.derived())
