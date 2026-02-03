from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trend_analysis import cli
from trend_analysis.api import RunResult
import trend_analysis.monte_carlo.runner as runner_module


def _write_scenario(path: Path, *, name: str = "mc_test", extra: str = "") -> None:
    payload = f"""
scenario:
  name: {name}
  description: Test scenario

base_config: config/defaults.yml

monte_carlo:
  mode: two_layer
  n_paths: 1
  horizon_years: 1.0
  frequency: M
  seed: 7
"""
    if extra:
        payload = payload + "\n" + extra
    path.write_text(payload.strip() + "\n", encoding="utf-8")


def _write_prices(path: Path) -> None:
    payload = "\n".join(
        [
            "Date,AssetA,AssetB",
            "2020-01-31,100,101",
            "2020-02-29,102,103",
            "2020-03-31,104,105",
            "2020-04-30,106,107",
        ]
    )
    path.write_text(payload + "\n", encoding="utf-8")


def test_mc_validate_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scenario_path = tmp_path / "scenario.yml"
    _write_scenario(scenario_path)

    rc = cli.main(["mc", "validate", str(scenario_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "mc_test" in out


def test_mc_validate_invalid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scenario_path = tmp_path / "invalid.yml"
    scenario_path.write_text(
        """
scenario:
  name: broken
monte_carlo:
  mode: two_layer
  n_paths: 1
  horizon_years: 1.0
  frequency: M
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rc = cli.main(["mc", "validate", str(scenario_path)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "base_config" in err


def test_mc_run_overrides_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario_path = tmp_path / "scenario.yml"
    data_path = tmp_path / "prices.csv"
    output_dir = tmp_path / "bundle"
    _write_scenario(scenario_path)
    _write_prices(data_path)

    def fake_run_simulation(*_args, **_kwargs) -> RunResult:
        metrics = pd.DataFrame({"sharpe_ratio": [1.23]}, index=["equal_weight"])
        return RunResult(metrics=metrics, details={}, seed=0, environment={})

    monkeypatch.setattr(runner_module, "run_simulation", fake_run_simulation)

    rc = cli.main(
        [
            "mc",
            "run",
            "--scenario",
            str(scenario_path),
            "--data",
            str(data_path),
            "--out",
            str(output_dir),
            "--n-paths",
            "3",
            "--jobs",
            "2",
            "--seed",
            "99",
        ]
    )

    assert rc == 0

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["settings"]["n_paths"] == 3
    assert manifest["settings"]["seed"] == 99
    assert manifest["settings"]["jobs"] == 2

    assert (output_dir / "results.csv").exists()
    assert (output_dir / "summary.csv").exists()


def test_mc_run_runtime_error_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scenario_path = tmp_path / "scenario.yml"
    data_path = tmp_path / "prices.csv"
    _write_scenario(scenario_path)
    _write_prices(data_path)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module.MonteCarloRunner, "run", _boom)

    rc = cli.main(
        [
            "mc",
            "run",
            "--scenario",
            str(scenario_path),
            "--data",
            str(data_path),
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "boom" in err
