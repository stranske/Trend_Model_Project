from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

import trend.cli as cli


def test_refresh_legacy_cli_module_updates_cache(monkeypatch):
    module = ModuleType("trend_analysis.cli")
    module.maybe_log_step = lambda *a, **k: None
    module._extract_cache_stats = lambda details: {"hits": 1}
    monkeypatch.setitem(sys.modules, "trend_analysis.cli", module)
    monkeypatch.setattr(cli, "_legacy_cli_module", None)
    monkeypatch.setattr(cli, "_legacy_maybe_log_step", cli._noop_maybe_log_step)
    monkeypatch.setattr(cli, "_legacy_extract_cache_stats", None)

    refreshed = cli._refresh_legacy_cli_module()

    assert refreshed is module
    assert cli._legacy_extract_cache_stats is module._extract_cache_stats
    assert cli._legacy_maybe_log_step is module.maybe_log_step


def test_run_pipeline_captures_portfolio_and_logging(monkeypatch, tmp_path):
    fake_returns = pd.DataFrame({"FundA": [0.01, 0.02]}, index=pd.RangeIndex(2))
    result = SimpleNamespace(
        details={
            "portfolio_user_weight": {"2024-01-31": 0.01, "2024-02-29": -0.005},
            "benchmarks": {"SPX": "benchmark"},
            "weights_user_weight": pd.DataFrame({"FundA": [0.6, 0.4]}),
        },
        metrics=pd.DataFrame({"Sharpe": [0.7]}),
    )

    monkeypatch.setattr(cli, "run_simulation", lambda cfg, df: result)

    class FakeRunLogging:
        @staticmethod
        def get_default_log_path(run_id: str) -> Path:
            return tmp_path / f"{run_id}.log"

        @staticmethod
        def init_run_logger(run_id: str, log_path: Path) -> None:
            log_path.touch()

    monkeypatch.setattr(cli, "run_logging", FakeRunLogging)

    steps: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        cli, "_legacy_maybe_log_step", lambda *a, **k: steps.append((a, k))
    )

    exports: list[tuple[bool, str]] = []
    monkeypatch.setattr(
        cli,
        "_handle_exports",
        lambda cfg, res, structured, run_id: exports.append((structured, run_id)),
    )

    bundles: list[Path] = []
    monkeypatch.setattr(
        cli,
        "_write_bundle",
        lambda cfg, res, source_path, bundle_path, structured, run_id: bundles.append(
            bundle_path
        ),
    )

    cfg = SimpleNamespace(export={}, sample_split={})
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    run_result, run_id, log_path = cli._run_pipeline(
        cfg,
        fake_returns,
        source_path=Path("returns.csv"),
        log_file=None,
        structured_log=True,
        bundle=bundle_dir,
    )

    assert run_result is result
    assert hasattr(result, "portfolio")
    assert hasattr(result, "benchmark") and result.benchmark == "benchmark"
    assert hasattr(result, "weights")
    assert log_path == tmp_path / f"{run_id}.log"
    assert exports == [(True, run_id)]
    assert bundles and bundles[0] == bundle_dir
    assert any(step[0][2] == "start" for step in steps)
    assert any(step[0][2] == "summary_render" for step in steps)


def test_handle_exports_excel_and_remaining(monkeypatch, tmp_path):
    export_calls: list[str] = []

    monkeypatch.setattr(
        cli.export, "make_summary_formatter", lambda *a, **k: "formatter"
    )
    monkeypatch.setattr(
        cli.export, "summary_frame_from_result", lambda details: {"rows": 1}
    )
    monkeypatch.setattr(
        cli.export,
        "export_to_excel",
        lambda data, path, default_sheet_formatter=None: export_calls.append("excel"),
    )
    monkeypatch.setattr(
        cli.export,
        "export_data",
        lambda data, path, formats: export_calls.append("data:" + ",".join(formats)),
    )
    monkeypatch.setattr(
        cli, "_legacy_maybe_log_step", lambda *a, **k: export_calls.append("log")
    )

    cfg = SimpleNamespace(
        export={
            "directory": str(tmp_path),
            "formats": ["xlsx", "csv"],
            "filename": "analysis",
        },
        sample_split={"in_start": "2020-01", "in_end": "2020-12"},
    )
    result = SimpleNamespace(metrics=pd.DataFrame({"Sharpe": [0.7]}), details={})

    cli._handle_exports(cfg, result, structured_log=False, run_id="run42")

    assert export_calls[:2] == ["excel", "data:csv"]
    assert export_calls[-1] == "log"


def test_write_bundle_into_directory(monkeypatch, tmp_path):
    bundle_dir = tmp_path / "out"
    bundle_dir.mkdir()
    recorded: list[Path] = []

    monkeypatch.setattr(
        "trend_analysis.export.bundle.export_bundle",
        lambda result, path: recorded.append(path),
    )
    monkeypatch.setattr(
        cli,
        "_legacy_maybe_log_step",
        lambda *a, **k: recorded.append(Path(k["bundle"])),
    )

    result = SimpleNamespace(details={}, metrics=pd.DataFrame())
    cli._write_bundle(
        SimpleNamespace(),
        result,
        source_path=Path("input.csv"),
        bundle_path=bundle_dir,
        structured_log=True,
        run_id="abc123",
    )

    assert recorded[0].name == "analysis_bundle.zip"
    assert getattr(result, "config") == {}
    assert getattr(result, "input_path") == Path("input.csv")


def test_print_summary_displays_cache_stats(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_legacy_extract_cache_stats", lambda details: {"hits": 3})
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "Summary")
    cfg = SimpleNamespace(sample_split={})
    result = SimpleNamespace(details={}, metrics=pd.DataFrame())

    cli._print_summary(cfg, result)
    captured = capsys.readouterr()
    assert "Summary" in captured.out
    assert "Cache statistics" in captured.out


def test_resolve_report_output_path_variants(tmp_path):
    export_dir = tmp_path / "reports"
    export_dir.mkdir()

    from_export_dir = cli._resolve_report_output_path(None, export_dir, "run7")
    assert from_export_dir.parent == export_dir
    assert from_export_dir.suffix == ".html"

    custom_html = cli._resolve_report_output_path("custom.html", None, "run7")
    assert custom_html.name == "custom.html"

    txt_path = cli._resolve_report_output_path("/tmp/report.txt", None, "run7")
    assert txt_path.suffix == ".txt"


def test_cli_entrypoint_invocation(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as exc:
        exec("raise SystemExit(main())", cli.__dict__)
    assert exc.value.code == 0
