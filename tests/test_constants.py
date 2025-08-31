"""Tests for constants module and their usage."""

from pathlib import Path

import pandas as pd
from trend_analysis import run_analysis, run_multi_analysis, cli
from trend_analysis.constants import DEFAULT_OUTPUT_DIRECTORY, DEFAULT_OUTPUT_FORMATS


def test_constants_exist():
    """Test that constants are properly defined."""
    assert DEFAULT_OUTPUT_DIRECTORY == "outputs"
    assert DEFAULT_OUTPUT_FORMATS == ["excel"]


def test_constants_are_used_in_run_analysis():
    """Test that run_analysis uses the constants."""
    # Check that the constants are imported and available in the module
    assert hasattr(run_analysis, "DEFAULT_OUTPUT_DIRECTORY")
    assert hasattr(run_analysis, "DEFAULT_OUTPUT_FORMATS")
    assert run_analysis.DEFAULT_OUTPUT_DIRECTORY == DEFAULT_OUTPUT_DIRECTORY
    assert run_analysis.DEFAULT_OUTPUT_FORMATS == DEFAULT_OUTPUT_FORMATS


def test_constants_are_used_in_run_multi_analysis():
    """Test that run_multi_analysis uses the constants."""
    # Check that the constants are imported and available in the module
    assert hasattr(run_multi_analysis, "DEFAULT_OUTPUT_DIRECTORY")
    assert hasattr(run_multi_analysis, "DEFAULT_OUTPUT_FORMATS")
    assert run_multi_analysis.DEFAULT_OUTPUT_DIRECTORY == DEFAULT_OUTPUT_DIRECTORY
    assert run_multi_analysis.DEFAULT_OUTPUT_FORMATS == DEFAULT_OUTPUT_FORMATS


def test_constants_are_used_in_cli():
    """Test that cli uses the constants."""
    # Check that the constants are imported and available in the module
    assert hasattr(cli, "DEFAULT_OUTPUT_DIRECTORY")
    assert hasattr(cli, "DEFAULT_OUTPUT_FORMATS")
    assert cli.DEFAULT_OUTPUT_DIRECTORY == DEFAULT_OUTPUT_DIRECTORY
    assert cli.DEFAULT_OUTPUT_FORMATS == DEFAULT_OUTPUT_FORMATS


def _write_cfg(path: Path, csv: Path) -> None:
    """Helper to write a minimal config file."""
    path.write_text(
        "\n".join(
            [
                "version: '1'",
                f"data: {{csv_path: '{csv}'}}",
                "preprocessing: {}",
                "vol_adjust: {target_vol: 1.0}",
                "sample_split: {in_start: '2020-01', in_end: '2020-03', "
                "out_start: '2020-04', out_end: '2020-06'}",
                "portfolio: {}",
                "metrics: {}",
                "export: {}",  # Empty export config to trigger defaults
                "run: {}",
            ]
        )
    )


def _make_df():
    """Helper to create a minimal test dataframe."""
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def test_constants_default_behavior_unchanged(tmp_path, monkeypatch):
    """Test that the default behavior using constants is unchanged."""
    csv = tmp_path / "data.csv"
    _make_df().to_csv(csv, index=False)
    cfg = tmp_path / "cfg.yml"
    _write_cfg(cfg, csv)
    monkeypatch.chdir(tmp_path)

    # Run analysis with empty export config (should use constants for defaults)
    rc = run_analysis.main(["-c", str(cfg)])
    assert rc == 0

    # Check that the default output directory and file are created
    assert (tmp_path / DEFAULT_OUTPUT_DIRECTORY / "analysis.xlsx").exists()
