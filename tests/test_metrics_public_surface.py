"""Contract tests for supported metrics and weighting package surfaces."""

from __future__ import annotations

import subprocess
import sys

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
    for name in metrics.__all__:
        assert hasattr(metrics, name), name


def test_metrics_package_has_no_compatibility_submodule_exports() -> None:
    probe = (
        "import trend_analysis.metrics as metrics\n"
        "for name in ('attribution', 'factor_attribution', 'rolling', 'summary', 'turnover'):\n"
        "    assert not hasattr(metrics, name), name\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
