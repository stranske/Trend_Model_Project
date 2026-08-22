"""Contract tests for supported metrics and weighting package surfaces."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import trend_analysis.metrics as metrics
import trend_analysis.weights as weights


def test_metrics_package_exposes_only_documented_metric_functions() -> None:
    assert set(metrics.__all__) == {
        "METRIC_REGISTRY",
        "alpha",
        "annual_return",
        "available_metrics",
        "deflated_sharpe_ratio",
        "estimate_sharpe_moments",
        "factor_exposures",
        "information_ratio",
        "max_drawdown",
        "probabilistic_sharpe_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "volatility",
    }
    for name in metrics.__all__:
        assert hasattr(metrics, name), name


def test_metrics_package_has_no_explicit_compatibility_bindings() -> None:
    retired = {"attribution", "factor_attribution", "rolling", "summary", "turnover"}
    assert retired.isdisjoint(metrics.__all__)

    module_path = Path(metrics.__file__).resolve()
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    assigned_names = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert retired.isdisjoint(assigned_names)
    assert "import_module" not in module_path.read_text(encoding="utf-8")


def test_metrics_submodules_remain_available_by_canonical_dotted_import() -> None:
    for name in ("attribution", "factor_attribution", "rolling", "summary", "turnover"):
        module = importlib.import_module(f"trend_analysis.metrics.{name}")
        assert module.__name__ == f"trend_analysis.metrics.{name}"


def test_weights_package_exports_the_intentional_algorithms_only() -> None:
    assert set(weights.__all__) == {
        "RiskParity",
        "HierarchicalRiskParity",
        "EqualRiskContribution",
        "EqualRiskContributionPolicy",
        "RobustMeanVariance",
        "RobustRiskParity",
    }
    for name in weights.__all__:
        assert hasattr(weights, name), name
