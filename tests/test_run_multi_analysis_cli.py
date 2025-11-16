from pathlib import Path

import pandas as pd

from trend_analysis import run_multi_analysis


def _write_cfg(path: Path, csv: Path, out_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "version: '1'",
                f"data: {{csv_path: '{csv}', date_column: 'Date', frequency: 'M'}}",
                "preprocessing: {}",
                "vol_adjust: {target_vol: 1.0}",
                "sample_split: {in_start: '2020-01', in_end: '2020-03', out_start: '2020-04', out_end: '2020-06'}",
                "multi_period: {frequency: M, in_sample_len: 2, out_sample_len: 1, start: '2020-01', end: '2020-03'}",
                "portfolio: {selection_mode: all, rebalance_calendar: NYSE, max_turnover: 0.5, transaction_cost_bps: 10}",
                "metrics: {}",
                f"export: {{directory: '{out_dir}', formats: ['csv']}}",
                "run: {}",
            ]
        )
    )


def _make_df():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    return pd.DataFrame({"Date": dates, "RF": 0.0, "A": 0.01})


def test_multi_cli_exports_files(tmp_path):
    csv = tmp_path / "data.csv"
    _make_df().to_csv(csv, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = tmp_path / "cfg.yml"
    _write_cfg(cfg, csv, out_dir)
    rc = run_multi_analysis.main(["-c", str(cfg)])
    assert rc == 0
    files = list(out_dir.glob("analysis_*.csv"))
    assert files, "no output files"


def test_multi_cli_detailed_output(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace

    from trend_analysis import run_multi_analysis

    cfg = SimpleNamespace(
        export={
            "directory": str(tmp_path),
            "formats": ["csv"],
            "filename": "custom",
        }
    )
    results = [
        {"period": ("2020-01", "2020-03", "2020-04", "2020-06"), "value": 1},
        {"period": ("2020-04", "2020-06", "2020-07", "2020-09"), "value": 2},
    ]
    summary = {"period": ("2020-01", "2020-03", "2020-07", "2020-09"), "value": 99}
    formatted: list[tuple[dict[str, object], tuple[str, ...]]] = []
    export_calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(run_multi_analysis, "load", lambda _: cfg)
    monkeypatch.setattr(run_multi_analysis, "run_mp", lambda _cfg: results)

    def fake_format(result: dict[str, object], *period: str) -> str:
        formatted.append((result, period))
        return f"summary-{result['value']}"

    monkeypatch.setattr(run_multi_analysis.export, "format_summary_text", fake_format)
    monkeypatch.setattr(
        run_multi_analysis.export,
        "combined_summary_result",
        lambda _results: summary,
    )

    def fake_export_phase1_multi_metrics(*args, **kwargs) -> None:
        export_calls.append(args)
        export_calls.append(tuple(sorted(kwargs.items())))

    monkeypatch.setattr(
        run_multi_analysis.export,
        "export_phase1_multi_metrics",
        fake_export_phase1_multi_metrics,
    )

    rc = run_multi_analysis.main(["--detailed", "-c", "dummy.yml"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "summary-1" in captured.out
    assert "summary-2" in captured.out
    assert "Combined Summary" in captured.out
    assert "summary-99" in captured.out
    err_lines = captured.err.strip().splitlines()
    assert err_lines, "expected logging output from setup_logging"
    assert any("Logging initialised" in line for line in err_lines)
    assert any("Log file initialised" in line for line in err_lines)

    assert formatted[0][0] == results[0]
    assert formatted[-1][0] == summary
    assert export_calls
    exported_args = export_calls[0]
    assert exported_args[0] == results
    assert exported_args[1].endswith("/custom")
    exported_kwargs = dict(export_calls[1])
    assert exported_kwargs == {"formats": ["csv"], "include_metrics": True}
