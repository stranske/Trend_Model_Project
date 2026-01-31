from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from trend_analysis.monte_carlo import MonteCarloScenario, MonteCarloSettings


def _load_example_payload() -> dict:
    root = Path(__file__).resolve().parents[2]
    scenario_path = root / "config" / "scenarios" / "monte_carlo" / "example.yml"

    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if "scenario" in payload:
        scenario_meta = payload.pop("scenario")
        payload = {**scenario_meta, **payload}
    return payload


def test_example_config_validates_against_schema() -> None:
    payload = _load_example_payload()
    scenario = MonteCarloScenario(**payload)

    assert scenario.name == "example_scenario"
    assert isinstance(scenario.monte_carlo, MonteCarloSettings)
    assert isinstance(scenario.monte_carlo.n_paths, int)
    assert scenario.monte_carlo.n_paths >= 1
    assert isinstance(scenario.monte_carlo.horizon_years, float)
    assert scenario.monte_carlo.horizon_years > 0.0
    assert scenario.monte_carlo.frequency in {"D", "W", "M", "Q", "Y"}
    assert isinstance(scenario.base_config, Path)
    assert scenario.return_model is not None
    assert isinstance(scenario.return_model, Mapping)
    assert scenario.folds is not None
    assert isinstance(scenario.folds, Mapping)
    assert scenario.outputs is not None
    assert isinstance(scenario.outputs, Mapping)
