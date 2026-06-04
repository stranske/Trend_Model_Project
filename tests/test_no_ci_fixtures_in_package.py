from __future__ import annotations

import importlib.machinery
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "trend_analysis._autofix_probe",
        "trend_analysis._autofix_trigger_sample",
        "trend_analysis._autofix_violation_case2",
        "trend_analysis._autofix_violation_case3",
        "trend_analysis._ci_probe_faults",
        "trend_analysis.automation_multifailure",
    ],
)
def test_ci_autofix_fixtures_are_not_packaged(module_name: str) -> None:
    package_dir = Path(__file__).resolve().parents[1] / "src" / "trend_analysis"
    module_leaf = module_name.rsplit(".", 1)[1]

    spec = importlib.machinery.PathFinder.find_spec(module_leaf, [str(package_dir)])

    assert spec is None
