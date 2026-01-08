from pathlib import Path

import pandas as pd

from trend_analysis import run_analysis


def _write_cfg(path: Path, csv: Path, out_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "version: '1'",
                f"data: {{csv_path: '{csv}', date_column: 'Date', frequency: 'M', risk_free_column: 'RF', allow_risk_free_fallback: false, missing_policy: drop}}",
                "preprocessing: {}",
                "vol_adjust: {target_vol: 1.0}",
                "sample_split: {in_start: '2020-01', in_end: '2020-03', "
                "out_start: '2020-04', out_end: '2020-06'}",
                "portfolio: {selection_mode: all, rebalance_calendar: NYSE, max_turnover: 0.5, transaction_cost_bps: 10, cost_model: {bps_per_trade: 0, slippage_bps: 0}}",
                "metrics: {}",
                f"export: {{directory: '{out_dir}', formats: ['csv']}}",
                "run: {}",
            ]
        )
    )


def _make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def test_cli_exports_files(tmp_path):
    csv = tmp_path / "data.csv"
    _make_df().to_csv(csv, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = tmp_path / "cfg.yml"
    _write_cfg(cfg, csv, out_dir)
    rc = run_analysis.main(["-c", str(cfg)])
    assert rc == 0
    assert (out_dir / "analysis_metrics.csv").exists()
