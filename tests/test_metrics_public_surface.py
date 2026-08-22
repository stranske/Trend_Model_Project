"""Contract tests for supported metrics and weighting package surfaces."""

from __future__ import annotations

import ast
from pathlib import Path

import trend_analysis.metrics as metrics
import trend_analysis.weights as weights


def test_metrics_package_exposes_only_documented_metric_functions() -> None:
    assert set(metrics.__all__) == {
        "METRIC_REGISTRY",
        "alpha",
        "annual_return",
        "annualize_return",
        "annualize_sharpe_ratio",
        "annualize_sortino_ratio",
        "annualize_volatility",
        "available_metrics",
        "deflated_sharpe_ratio",
        "estimate_sharpe_moments",
        "factor_exposures",
        "info_ratio",
        "information_ratio",
        "max_drawdown",
        "probabilistic_sharpe_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "volatility",
    }


def test_metrics_package_has_no_compatibility_submodule_exports() -> None:
    source = Path(metrics.__file__).read_text()
    tree = ast.parse(source)
    compatibility_names = {
        "attribution",
        "factor_attribution",
        "rolling",
        "summary",
        "turnover",
    }
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert compatibility_names.isdisjoint(assigned_names)


def test_weights_package_exports_the_intentional_algorithms_only() -> None:
    assert set(weights.__all__) == {
        "RiskParity",
        "HierarchicalRiskParity",
        "EqualRiskContribution",
        "EqualRiskContributionPolicy",
        "RobustMeanVariance",
        "RobustRiskParity",
    }
