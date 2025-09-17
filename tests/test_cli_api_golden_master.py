from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import yaml  # type: ignore[import-untyped]

from trend_analysis import api, export
from trend_analysis.cli import run_cli_simulation_from_paths
from trend_analysis.config import Config


def make_test_data() -> pd.DataFrame:
    """Create deterministic monthly return data."""

    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    return pd.DataFrame(
        {
            "Date": dates,
            "RF": 0.0,
            "Fund_A": [0.01 + 0.001 * i for i in range(len(dates))],
            "Fund_B": [0.015 + 0.002 * i for i in range(len(dates))],
            "Fund_C": [0.008 + 0.0005 * i for i in range(len(dates))],
            "SPX": [0.012 + 0.001 * i for i in range(len(dates))],
        }
    )


def make_test_config(csv_path: str, *, seed: int = 42) -> Config:
    """Build a minimal configuration for the comparison test."""

    return Config(
        version="1",
        data={"csv_path": csv_path},
        preprocessing={},
        vol_adjust={"target_vol": 1.0},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        portfolio={},
        benchmarks={"spx": "SPX"},
        metrics={},
        export={},
        run={},
        seed=seed,
    )


def _write_config(cfg_path: Path, config: Config) -> None:
    """Persist the Config object to YAML for CLI consumption."""

    config_dict = config.model_dump()
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_dict, fh, default_flow_style=False, sort_keys=False)


def test_cli_matches_api_output() -> None:
    """CLI helper should match API metrics and summary output."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "test_data.csv"
        config_file = tmp_path / "test_config.yml"

        df = make_test_data()
        df.to_csv(csv_file, index=False)

        cfg = make_test_config(str(csv_file), seed=1234)
        _write_config(config_file, cfg)

        api_result = api.run_simulation(cfg, df)
        cli_result, cli_config = run_cli_simulation_from_paths(
            str(config_file), str(csv_file)
        )

        pd.testing.assert_frame_equal(cli_result.metrics, api_result.metrics)
        assert cli_result.seed == api_result.seed
        assert cli_result.environment == api_result.environment

        split = cli_config.sample_split
        summary_cli = export.format_summary_text(
            cli_result.details,
            str(split.get("in_start")),
            str(split.get("in_end")),
            str(split.get("out_start")),
            str(split.get("out_end")),
        )
        summary_api = export.format_summary_text(
            api_result.details,
            str(split.get("in_start")),
            str(split.get("in_end")),
            str(split.get("out_start")),
            str(split.get("out_end")),
        )
        assert summary_cli == summary_api
