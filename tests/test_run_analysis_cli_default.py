import pandas as pd
from pathlib import Path

from trend_analysis import run_analysis


def _write_cfg(path: Path, csv: Path, out_dir: Path) -> None:
    """Write a minimal config file for the CLI test.

    The default behaviour of ``run_analysis`` is to emit output files to a
    directory named ``outputs`` in the current working directory.  On CI this
    can lead to permission errors when the repository root is read-only.  By
    explicitly writing the output to a temporary directory provided by the
    ``tmp_path`` fixture we ensure the test always has write access.
    """

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
                f"export: {{directory: '{out_dir}', formats: []}}",
                "run: {}",
            ]
        )
    )


def _make_df():
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def test_cli_default_output(tmp_path, capsys):
    csv = tmp_path / "data.csv"
    _make_df().to_csv(csv, index=False)
    cfg = tmp_path / "cfg.yml"

    # Use a dedicated output directory within ``tmp_path`` to guarantee
    # writable permissions and isolate test artefacts.
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    _write_cfg(cfg, csv, out_dir)

    rc = run_analysis.main(["-c", str(cfg)])
    captured = capsys.readouterr().out
    assert rc == 0
    assert "Vol-Adj Trend Analysis" in captured
