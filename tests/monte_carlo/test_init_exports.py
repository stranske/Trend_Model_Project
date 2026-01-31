"""Tests for monte_carlo package exports."""

from __future__ import annotations

from trend_analysis import monte_carlo


def test_monte_carlo_exports_include_schema_types() -> None:
    assert "MonteCarloScenario" in monte_carlo.__all__
    assert "MonteCarloSettings" in monte_carlo.__all__

    # Ensure module-level attributes are the expected types.
    scenario_cls = monte_carlo.MonteCarloScenario
    settings_cls = monte_carlo.MonteCarloSettings
    assert scenario_cls.__name__ == "MonteCarloScenario"
    assert settings_cls.__name__ == "MonteCarloSettings"
