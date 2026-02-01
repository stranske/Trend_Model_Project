from __future__ import annotations

import trend_analysis.monte_carlo as monte_carlo
from trend_analysis.monte_carlo import MonteCarloScenario, MonteCarloSettings


def test_module_exports_monte_carlo_schemas() -> None:
    assert "MonteCarloScenario" in monte_carlo.__all__
    assert "MonteCarloSettings" in monte_carlo.__all__
    assert monte_carlo.MonteCarloScenario is MonteCarloScenario
    assert monte_carlo.MonteCarloSettings is MonteCarloSettings
