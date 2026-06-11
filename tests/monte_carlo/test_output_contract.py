from __future__ import annotations

import json

import pandas as pd

from trend_analysis.cli import _write_mc_manifest
from trend_analysis.monte_carlo.aggregator import aggregate_monte_carlo_results
from trend_analysis.monte_carlo.export import export_aggregation_results
from trend_analysis.monte_carlo.results import (
    MonteCarloResults,
    build_summary_frame,
    export_results,
)
from trend_analysis.monte_carlo.scenario import MonteCarloScenario, MonteCarloSettings


def test_export_results_writes_flat_bundle_contract(tmp_path) -> None:
    results_frame = pd.DataFrame(
        [
            {
                "fold_id": 1,
                "path_id": 1,
                "strategy": "rank_12_equal",
                "metric": 0.42,
            }
        ]
    )
    results = MonteCarloResults(
        mode="two_layer",
        evaluations=[],
        errors=[],
        results_frame=results_frame,
        summary_frame=build_summary_frame(results_frame),
        metadata={},
    )

    exported = export_results(results, tmp_path, formats=["csv"])

    assert exported == {
        "results_csv": tmp_path / "results.csv",
        "summary_csv": tmp_path / "summary.csv",
    }
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "config_snapshot.yml").exists()
    assert not (tmp_path / "distributions").exists()
    assert not (tmp_path / "paths").exists()
    assert not (tmp_path / "strategies").exists()
    assert not (tmp_path / "logs").exists()


def test_mc_manifest_indexes_cli_exported_results_only(tmp_path) -> None:
    results_frame = pd.DataFrame(
        [
            {
                "fold_id": 1,
                "path_id": 1,
                "strategy": "rank_12_equal",
                "metric": 0.42,
            }
        ]
    )
    results = MonteCarloResults(
        mode="two_layer",
        evaluations=[],
        errors=[],
        results_frame=results_frame,
        summary_frame=build_summary_frame(results_frame),
        metadata={},
    )
    exported = export_results(results, tmp_path, formats=["csv"])
    scenario = MonteCarloScenario(
        name="flat_bundle_contract",
        base_config="config/demo.yml",
        monte_carlo=MonteCarloSettings(
            mode="two_layer",
            n_paths=1,
            horizon_years=1.0,
            frequency="M",
        ),
    )

    manifest_path = _write_mc_manifest(
        tmp_path,
        scenario=scenario,
        results=results,
        overrides={"n_paths": 1},
        exported_files=exported,
        data_path=None,
        jobs_used=1,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["outputs"]["files"] == {
        "results_csv": str(tmp_path / "results.csv"),
        "summary_csv": str(tmp_path / "summary.csv"),
    }
    assert "path_summary_csv" not in manifest["outputs"]["files"]


def test_export_aggregation_results_writes_flat_bundle_contract(tmp_path) -> None:
    results_frame = pd.DataFrame(
        [
            {"path_id": 1, "strategy": "rank_12_equal", "metric": 0.42},
            {"path_id": 2, "strategy": "rank_12_equal", "metric": 0.84},
        ]
    )
    aggregation = aggregate_monte_carlo_results(
        results_frame,
        quantiles=[0.5],
        breach_spec={"metric": [0.5]},
        expected_shortfall_spec={"metric": 0.5},
    )

    exported = export_aggregation_results(aggregation, tmp_path, formats=["csv"])

    assert exported == {
        "path_summary_csv": tmp_path / "path_summary.csv",
        "per_strategy_stats_csv": tmp_path / "per_strategy_stats.csv",
        "per_strategy_path_csv": tmp_path / "per_strategy_path.csv",
        "quantiles_csv": tmp_path / "quantiles.csv",
        "summary_quantiles_csv": tmp_path / "summary_quantiles.csv",
        "breach_probabilities_csv": tmp_path / "breach_probabilities.csv",
        "expected_shortfall_csv": tmp_path / "expected_shortfall.csv",
    }
    for filename in (
        "path_summary.csv",
        "per_strategy_stats.csv",
        "per_strategy_path.csv",
        "quantiles.csv",
        "summary_quantiles.csv",
        "breach_probabilities.csv",
        "expected_shortfall.csv",
    ):
        assert (tmp_path / filename).exists()
    assert not (tmp_path / "distributions").exists()
    assert not (tmp_path / "paths").exists()
    assert not (tmp_path / "strategies").exists()
