from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import trend.cli as cli

# Pre-import the bundle module at collection time to avoid lazy-loading race
# conditions when pytest-xdist runs tests in parallel. This ensures the module
# is in sys.modules before workers fork and prevents ImportError when
# monkeypatching trend_analysis.export.bundle in test_write_bundle_into_directory.
import trend_analysis.export.bundle as _bundle_mod  # noqa: F401


def test_cli_uses_canonical_logging_and_cache_helpers() -> None:
    assert callable(cli.maybe_log_step)
    assert callable(cli.extract_cache_stats)


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
    monkeypatch.chdir(tmp_path)

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
    monkeypatch.setattr(cli, "maybe_log_step", lambda *a, **k: steps.append((a, k)))

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
        lambda cfg, res, source_path, bundle_path, structured, run_id: bundles.append(bundle_path),
    )

    cfg = SimpleNamespace(
        export={},
        sample_split={},
        portfolio={"cost_model": {"per_trade_bps": 12.0, "half_spread_bps": 0}},
    )
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

    monkeypatch.setattr(cli.export, "make_summary_formatter", lambda *a, **k: "formatter")
    monkeypatch.setattr(cli.export, "summary_frame_from_result", lambda details: {"rows": 1})
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
    monkeypatch.setattr(cli, "maybe_log_step", lambda *a, **k: export_calls.append("log"))

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

    assert export_calls == ["log", "excel", "data:csv", "log"]


def test_write_run_artifacts_emits_replayable_envelope(monkeypatch, tmp_path):
    manifest_dir = tmp_path / "runs" / "run42"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("version: '1'\n", encoding="utf-8")
    input_path = tmp_path / "returns.csv"
    input_path.write_text("Date,A\n2020-01-31,0.1\n", encoding="utf-8")
    result = SimpleNamespace(details=None, metrics=pd.DataFrame({"Sharpe": [0.7]}))
    cfg = SimpleNamespace(
        export={"directory": str(tmp_path), "formats": ["csv"], "filename": "analysis"},
        sample_split={},
        model_dump=lambda: {"version": "1"},
    )
    recorded: dict[str, object] = {}
    log_events: list[str] = []

    monkeypatch.setattr(cli, "write_run_artifacts", lambda **_kwargs: manifest_dir)
    monkeypatch.setattr(cli.IdentityMap, "from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *_args, **_kwargs: "summary")

    def fake_write_run_envelope(run_result, **kwargs):
        recorded["result"] = run_result
        recorded.update(kwargs)
        return manifest_dir / "run_envelope.json"

    monkeypatch.setattr(cli, "write_run_envelope", fake_write_run_envelope)
    monkeypatch.setattr(
        cli,
        "maybe_log_step",
        lambda _enabled, _run_id, event, _message, **_fields: log_events.append(event),
    )

    written = cli._write_trend_run_artifacts(
        cfg=cfg,
        result=result,
        config_path=config_path,
        input_path=input_path,
        data_frame=pd.DataFrame({"A": [0.1]}),
        run_id="run42",
        structured_log=True,
    )

    assert written == manifest_dir
    assert recorded["result"] is result
    assert recorded["manifest_path"] == manifest_dir / "manifest.json"
    assert recorded["run_dir"] == manifest_dir
    assert recorded["config"] == {"version": "1"}
    assert log_events == ["run_artifacts", "run_envelope"]


def test_write_run_artifacts_survives_envelope_failure(monkeypatch, tmp_path):
    manifest_dir = tmp_path / "runs" / "run42"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.yml"
    config_path.write_text("version: '1'\n", encoding="utf-8")
    input_path = tmp_path / "returns.csv"
    result = SimpleNamespace(details=None, metrics=pd.DataFrame({"Sharpe": [0.7]}))
    cfg = SimpleNamespace(
        export={"directory": str(tmp_path), "formats": ["csv"], "filename": "analysis"},
        sample_split={},
        model_dump=lambda: {"version": "1"},
    )
    log_events: list[str] = []

    monkeypatch.setattr(cli, "write_run_artifacts", lambda **_kwargs: manifest_dir)
    monkeypatch.setattr(cli.IdentityMap, "from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *_args, **_kwargs: "summary")
    monkeypatch.setattr(
        cli,
        "write_run_envelope",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("envelope failed")),
    )
    monkeypatch.setattr(
        cli,
        "maybe_log_step",
        lambda _enabled, _run_id, event, _message, **_fields: log_events.append(event),
    )

    written = cli._write_trend_run_artifacts(
        cfg=cfg,
        result=result,
        config_path=config_path,
        input_path=input_path,
        data_frame=pd.DataFrame({"A": [0.1]}),
        run_id="run42",
        structured_log=True,
    )

    assert written == manifest_dir
    assert log_events == ["run_artifacts"]


def test_write_bundle_into_directory(monkeypatch, tmp_path):
    bundle_dir = tmp_path / "out"
    bundle_dir.mkdir()
    recorded: list[Path] = []

    # Use the pre-imported module from the top of the file to avoid
    # lazy-loading race conditions in parallel test execution.
    monkeypatch.setattr(
        _bundle_mod,
        "export_bundle",
        lambda result, path: recorded.append(path),
    )
    monkeypatch.setattr(
        cli,
        "maybe_log_step",
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
    monkeypatch.setattr(cli, "extract_cache_stats", lambda details: {"hits": 3})
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
