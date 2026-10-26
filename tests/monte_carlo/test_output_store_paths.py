"""Reject path-retention knobs that the exporter cannot honor."""

from pathlib import Path

import pytest
import yaml

from trend_analysis.monte_carlo import MonteCarloScenario
from trend_analysis.monte_carlo.registry import load_scenario_from_path


def _scenario(outputs: object) -> dict:
    return {
        "name": "path_retention",
        "base_config": "config/defaults.yml",
        "monte_carlo": {
            "mode": "two_layer",
            "n_paths": 2,
            "horizon_years": 1,
            "frequency": "M",
        },
        "outputs": outputs,
    }


@pytest.mark.parametrize(
    "store_paths",
    [{"nav": 50}, {"nav": 0}, {"weights": 2}, {"weights": 0}, {}, None],
)
def test_store_paths_is_rejected_before_export(tmp_path: Path, store_paths: object) -> None:
    """The real YAML load must refuse both positive and zero retention limits."""
    scenario_path = tmp_path / "scenario.yml"
    output_dir = tmp_path / "export"
    scenario_path.write_text(
        yaml.safe_dump(_scenario({"directory": str(output_dir), "store_paths": store_paths})),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"outputs\.store_paths.*not supported"):
        load_scenario_from_path(scenario_path)
    assert not output_dir.exists()


def test_store_paths_is_rejected_without_output_directory() -> None:
    with pytest.raises(ValueError, match=r"outputs\.store_paths.*not supported"):
        MonteCarloScenario(**_scenario({"store_paths": {"nav": 50}}))


@pytest.mark.parametrize("outputs", [None, {}, {"directory": "out", "formats": ["csv"]}])
def test_supported_outputs_still_load(outputs: object) -> None:
    assert MonteCarloScenario(**_scenario(outputs)).outputs == outputs
