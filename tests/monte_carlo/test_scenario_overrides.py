from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from trend_analysis.monte_carlo.registry import load_scenario_from_path
from trend_analysis.monte_carlo.runner import MonteCarloRunner


def _scenario(tmp_path: Path, overrides: dict) -> MonteCarloRunner:
    path = tmp_path / "scenario.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "scenario": {"name": "section-overrides"},
                "base_config": str(Path("config/defaults.yml").resolve()),
                "monte_carlo": {
                    "mode": "two_layer",
                    "n_paths": 1,
                    "horizon_years": 1,
                    "frequency": "M",
                    "seed": 7,
                },
                **overrides,
            }
        ),
        encoding="utf-8",
    )
    return MonteCarloRunner(load_scenario_from_path(path))


def test_scenario_multi_period_override_reaches_base_config(tmp_path: Path) -> None:
    runner = _scenario(tmp_path, {"multi_period": {"min_funds": 99}})
    assert runner.base_config["multi_period"]["min_funds"] == 99
    assert runner.base_config["multi_period"]["max_funds"] == 25


@pytest.mark.parametrize(
    ("section", "override"),
    [
        ("signals", {"window": 27}),
        ("robustness", {"bootstrap_samples": 17}),
        ("performance", {"enable_cache": False}),
    ],
)
def test_scenario_pipeline_sections_reach_base_config(
    tmp_path: Path, section: str, override: dict
) -> None:
    runner = _scenario(tmp_path, {section: override})
    for key, value in override.items():
        assert runner.base_config[section][key] == value


def test_scenario_only_sections_do_not_leak_into_pipeline_config(tmp_path: Path) -> None:
    runner = _scenario(tmp_path, {"outputs": {"formats": ["csv"]}})
    assert "outputs" not in runner.base_config
    assert "monte_carlo" not in runner.base_config


def test_cli_manifest_records_effective_scenario_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trend import cli
    from trend_analysis.api import RunResult
    import pandas as pd

    runner = _scenario(
        tmp_path,
        {
            "multi_period": {"min_funds": 2},
            "metrics": {"rf_override_enabled": True, "rf_rate_annual": 0.0},
            "signals": {"window": 27},
            "performance": {"enable_cache": False},
        },
    )
    used_configs = []

    def run_simulation(config, *_args, **_kwargs):
        used_configs.append(config)
        return RunResult(
            metrics=pd.DataFrame({"sharpe_ratio": [1.23]}, index=["equal_weight"]),
            details={},
            seed=0,
            environment={},
        )

    monkeypatch.setattr("trend_analysis.monte_carlo.runner.run_simulation", run_simulation)
    data = tmp_path / "prices.csv"
    data.write_text(
        "Date,AssetA,AssetB\n2020-01-31,100,101\n2020-02-29,102,103\n"
        "2020-03-31,104,105\n2020-04-30,106,107\n",
        encoding="utf-8",
    )
    output = tmp_path / "bundle"
    assert (
        cli.main(
            [
                "mc",
                "run",
                "--scenario",
                str(tmp_path / "scenario.yml"),
                "--data",
                str(data),
                "--out",
                str(output),
                "--n-paths",
                "2",
                "--jobs",
                "1",
                "--seed",
                "99",
                "--no-progress",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    effective = manifest["effective_config_sections"]
    assert effective["multi_period"] == runner.base_config["multi_period"]
    assert effective["multi_period"]["min_funds"] == 2
    assert effective["multi_period"]["max_funds"] == 25
    assert effective["signals"] == runner.base_config["signals"]
    assert effective["signals"]["window"] == 27
    assert effective["metrics"] == runner.base_config["metrics"]
    assert effective["metrics"]["rf_override_enabled"] is True
    assert effective["performance"]["enable_cache"] is False
    assert "monte_carlo" not in effective
    assert manifest["overrides"] == {"n_paths": 2, "jobs": 1, "seed": 99}
    assert manifest["settings"]["n_paths"] == 2
    assert manifest["settings"]["jobs"] == 1
    assert manifest["settings"]["seed"] == 99
    assert manifest["results"]["errors"] == 0
    assert len(used_configs) == 2
    for config in used_configs:
        assert config.metrics == effective["metrics"]
        assert config.performance == effective["performance"]


def test_cli_manifest_serialises_unquoted_yaml_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import date

    from trend import cli
    from trend_analysis.api import RunResult
    import pandas as pd

    base_config = Path("config/defaults.yml").resolve()
    scenario_path = tmp_path / "scenario.yml"
    scenario_path.write_text(
        "\n".join(
            [
                "scenario:",
                "  name: yaml-date-manifest",
                f"base_config: {base_config}",
                "monte_carlo:",
                "  mode: two_layer",
                "  n_paths: 1",
                "  horizon_years: 1",
                "  frequency: M",
                "  seed: 7",
                "sample_split:",
                "  date: 2017-12-31",
            ]
        ),
        encoding="utf-8",
    )
    runner = MonteCarloRunner(load_scenario_from_path(scenario_path))
    assert runner.base_config["sample_split"]["date"] == date(2017, 12, 31)

    def run_simulation(config, *_args, **_kwargs):
        return RunResult(
            metrics=pd.DataFrame({"sharpe_ratio": [1.0]}, index=["equal_weight"]),
            details={},
            seed=0,
            environment={},
        )

    monkeypatch.setattr("trend_analysis.monte_carlo.runner.run_simulation", run_simulation)
    data = tmp_path / "prices.csv"
    data.write_text(
        "Date,AssetA,AssetB\n2020-01-31,100,101\n2020-02-29,102,103\n"
        "2020-03-31,104,105\n2020-04-30,106,107\n",
        encoding="utf-8",
    )
    output = tmp_path / "bundle"
    assert (
        cli.main(
            [
                "mc",
                "run",
                "--scenario",
                str(scenario_path),
                "--data",
                str(data),
                "--out",
                str(output),
                "--no-progress",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["effective_config_sections"]["sample_split"]["date"] == "2017-12-31"
