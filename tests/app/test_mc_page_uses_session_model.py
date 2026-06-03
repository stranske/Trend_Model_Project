from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.streamlit.test_mc_page import _load_page, _make_scenario
from trend_analysis.monte_carlo.registry import ScenarioRegistryEntry
from trend_analysis.monte_carlo.scenario import MonteCarloScenario


def test_mc_page_uses_session_model_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    page, stub = _load_page(monkeypatch)

    monkeypatch.setattr(
        page,
        "list_scenarios",
        lambda **_kwargs: [
            ScenarioRegistryEntry(
                name="macro",
                path=Path("config/scenarios/monte_carlo/example.yml"),
                description="Macro scenario",
                tags=("macro",),
            )
        ],
    )
    monkeypatch.setattr(page, "load_scenario", lambda name: _make_scenario(name))

    dates = pd.date_range("2022-01-31", periods=6, freq="ME")
    returns = pd.DataFrame(
        {
            "FundA": [0.01, 0.02, -0.01, 0.03, 0.01, 0.00],
            "FundB": [0.00, 0.01, 0.02, -0.01, 0.01, 0.02],
            "Bench": [0.005, 0.006, 0.004, 0.007, 0.005, 0.006],
        },
        index=dates,
    )
    stub.session_state.update(
        {
            "returns_df": returns,
            "schema_meta": {"frequency": "M"},
            "upload_status": "success",
            "model_state": {
                "metric_weights": {"sharpe": 1.0},
                "selection_count": 2,
                "multi_period_frequency": "M",
                "lookback_periods": 3,
                "evaluation_periods": 1,
            },
            "analysis_fund_columns": ["FundA", "FundB", "Bench"],
            "selected_benchmark": "Bench",
        }
    )

    captured: dict[str, Any] = {}

    class FakeRunner:
        def __init__(self, scenario: MonteCarloScenario, **kwargs: Any) -> None:
            captured["scenario"] = scenario
            captured["kwargs"] = kwargs

        def validate(self) -> list[str]:
            return []

    monkeypatch.setattr(page, "MonteCarloRunner", FakeRunner)
    stub.button_responses = [False, True]

    page.render()

    kwargs = captured["kwargs"]
    assert "base_config" in kwargs
    assert "price_history" in kwargs

    price_history = kwargs["price_history"]
    assert list(price_history.columns) == ["FundA", "FundB", "Bench"]
    assert price_history.loc[dates[0], "FundA"] == pytest.approx(101.0)
    assert price_history.loc[dates[1], "FundA"] == pytest.approx(103.02)
    assert kwargs["base_config"].portfolio["rank"]["n"] == 2
    assert kwargs["base_config"].benchmarks == {"Bench": "Bench"}
    assert kwargs["base_config"].portfolio["indices_list"] == ["Bench"]
