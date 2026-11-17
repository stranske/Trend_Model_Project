from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from trend_analysis.monte_carlo.registry import load_scenario
from trend_analysis.regimes import compute_regimes, normalise_settings


def test_cost_regime_example_loads_without_exception() -> None:
    # Repository-supported invocation:
    # python -m pytest tests/monte_carlo/scenario/test_cost_regime_example_load.py -q --no-cov
    scenario = load_scenario("cost_regime_example")

    assert scenario.name == "cost_regime_example"
    assert scenario.costs is not None
    assert isinstance(scenario.costs, Mapping)
    assert scenario.costs.get("kind") == "regime_stochastic"
    raw = scenario.raw
    assert isinstance(raw, Mapping)
    regime = raw.get("regime")
    assert isinstance(regime, Mapping)
    assert regime.get("enabled") is True
    assert regime.get("proxy") == "AssetA"


def test_cost_regime_example_regime_proxy_yields_risk_on_and_risk_off() -> None:
    scenario = load_scenario("cost_regime_example")
    raw = scenario.raw
    assert isinstance(raw, Mapping)
    regime_cfg = raw.get("regime")
    assert isinstance(regime_cfg, Mapping)
    settings = normalise_settings(regime_cfg)

    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    asset_levels = 100.0 * np.cumprod(1.0 + np.resize([0.01, -0.005, 0.02, -0.01], len(dates)))
    # Regime rolling_return expects period returns, not price levels.
    proxy_returns = pd.Series(asset_levels, index=dates).pct_change().fillna(0.0)
    proxy = proxy_returns.rename(str(settings.proxy))

    labels = compute_regimes(proxy, settings, freq="M", periods_per_year=12)
    observed = set(labels.dropna().astype("string"))
    assert {"Risk-On", "Risk-Off"}.issubset(observed)
