from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import trend.cli as cli
from trend.cli import (
    SCENARIO_WINDOWS,
    TrendCLIError,
    _adjust_for_scenario,
    _determine_seed,
    _resolve_returns_path,
    build_parser,
    main,
)
from trend_analysis.api import RunResult
from trend_analysis.reporting import generate_unified_report


def _sample_result() -> RunResult:
    metrics = pd.DataFrame({"Sharpe": [0.82], "CAGR": [0.11]}, index=["FundA"])
    turnover = pd.Series(
        [0.18, 0.22, 0.2],
        index=pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"]),
    )
    final_weights = pd.Series({"FundA": 0.55, "FundB": 0.45})
    portfolio = pd.Series(
        [0.012, -0.004, 0.01],
        index=pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"]),
    )
    stats = SimpleNamespace(
        cagr=0.11,
        vol=0.08,
        sharpe=1.05,
        sortino=0.9,
        max_drawdown=-0.18,
        information_ratio=0.35,
    )
    details = {
        "out_user_stats": stats,
        "out_ew_stats": stats,
        "selected_funds": ["FundA", "FundB"],
        "risk_diagnostics": {
            "turnover": turnover,
            "final_weights": final_weights,
        },
    }
    result = RunResult(metrics=metrics, details=details, seed=13, environment={})
    setattr(result, "portfolio", portfolio)
    return result


def _sample_config() -> SimpleNamespace:
    return SimpleNamespace(
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        vol_adjust={"target_vol": 0.15},
        portfolio={"selection_mode": "all", "weighting_scheme": "equal"},
        run={},
        benchmarks={},
    )


def test_build_parser_contains_expected_subcommands() -> None:
    parser = build_parser()
    expected_subcommands = {"run", "report", "stress", "app"}
    for subcommand in expected_subcommands:
        # Should not raise SystemExit
        try:
            args = parser.parse_args([subcommand])
        except SystemExit:
            assert False, f"Subcommand '{subcommand}' not recognized by parser"
        # The subcommand should be set in the namespace
        assert getattr(args, "subcommand", None) == subcommand


def test_legacy_callable_returns_fallback_when_module_missing(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_legacy_cli_module", None)
    monkeypatch.setattr(cli, "_refresh_legacy_cli_module", lambda: None)

    def sentinel() -> str:
        return "fallback"

    resolved = cli._legacy_callable("missing", sentinel)

    assert resolved is sentinel


def test_resolve_returns_path_uses_config_directory(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("", encoding="utf-8")

    class DummyCfg:
        data = {"csv_path": "data/returns.csv"}

    resolved = _resolve_returns_path(cfg_path, DummyCfg(), None)
    assert resolved == (tmp_path / "data" / "returns.csv").resolve()


def test_resolve_returns_path_falls_back_to_parent_directory(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "demo.yml"
    cfg_path.write_text("", encoding="utf-8")

    data_dir = tmp_path / "demo"
    data_dir.mkdir()
    target = data_dir / "demo_returns.csv"
    target.write_text("Date,Mgr_01\n2020-01-31,0.01\n", encoding="utf-8")

    class DummyCfg:
        data = {"csv_path": "demo/demo_returns.csv"}

    resolved = _resolve_returns_path(cfg_path, DummyCfg(), None)
    assert resolved == target.resolve()


def test_resolve_returns_path_requires_csv(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("", encoding="utf-8")

    class DummyCfg:
        data: dict[str, str] = {}

    with pytest.raises(TrendCLIError):
        _resolve_returns_path(cfg_path, DummyCfg(), None)


def test_determine_seed_precedence(monkeypatch) -> None:
    class DummyCfg:
        seed = 7

    cfg = DummyCfg()
    assert _determine_seed(cfg, 21) == 21
    monkeypatch.setenv("TREND_SEED", "33")
    cfg_env = DummyCfg()
    assert _determine_seed(cfg_env, None) == 33
    monkeypatch.delenv("TREND_SEED")
    cfg_default = DummyCfg()
    assert _determine_seed(cfg_default, None) == 7


def test_determine_seed_handles_invalid_env(monkeypatch) -> None:
    class DummyCfg:
        seed = 11

    cfg = DummyCfg()
    monkeypatch.setenv("TREND_SEED", "not-an-int")
    value = _determine_seed(cfg, None)

    assert value == 11
    assert cfg.seed == 11


def test_adjust_for_scenario_updates_sample_split() -> None:
    class DummyCfg:
        sample_split = {}

    cfg = DummyCfg()
    _adjust_for_scenario(cfg, "2008")
    assert cfg.sample_split["in_start"] == SCENARIO_WINDOWS["2008"][0][0]
    assert cfg.sample_split["out_end"] == SCENARIO_WINDOWS["2008"][1][1]


def test_main_run_invokes_pipeline(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,Mgr_01\n2020-01-31,0.01\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_ensure_dataframe(path: Path) -> pd.DataFrame:
        captured["returns_path"] = path
        return pd.DataFrame({"Date": ["2020-01-31"], "Mgr_01": [0.01]})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        captured["pipeline_args"] = kwargs
        return RunResult(pd.DataFrame(), {}, 42, {}), "run123", Path("log.json")

    monkeypatch.setattr("trend.cli._ensure_dataframe", fake_ensure_dataframe)
    config_obj = SimpleNamespace(
        sample_split={},
        vol_adjust={},
        portfolio={},
        run={},
        benchmarks={},
        export={},
    )
    monkeypatch.setattr(cli, "load_config", lambda path: config_obj)
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)

    exit_code = main(["run", "--config", "config/demo.yml", "--returns", str(csv_path)])

    assert exit_code == 0
    assert captured["returns_path"] == csv_path.resolve()
    pipeline_kwargs = captured["pipeline_args"]
    assert pipeline_kwargs["source_path"] == csv_path.resolve()
    assert pipeline_kwargs["structured_log"] is True


def test_main_run_without_structured_log(monkeypatch, tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("Date,Mgr_01\n2020-01-31,0.01\n", encoding="utf-8")

    def fake_ensure_dataframe(path: Path) -> pd.DataFrame:
        assert path == csv_path.resolve()
        return pd.DataFrame({"Date": ["2020-01-31"], "Mgr_01": [0.01]})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        assert kwargs["structured_log"] is False
        return RunResult(pd.DataFrame(), {}, 42, {}), "runXYZ", None

    monkeypatch.setattr("trend.cli._ensure_dataframe", fake_ensure_dataframe)
    config_obj = SimpleNamespace(
        sample_split={},
        vol_adjust={},
        portfolio={},
        run={},
        benchmarks={},
        export={},
    )
    monkeypatch.setattr(cli, "load_config", lambda path: config_obj)
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)

    exit_code = main(
        [
            "run",
            "--config",
            "config/demo.yml",
            "--returns",
            str(csv_path),
            "--no-structured-log",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Structured log" not in out


def test_main_run_requires_config() -> None:
    exit_code = main(["run"])
    assert exit_code == 2


def test_main_report_uses_requested_directory(monkeypatch, tmp_path: Path) -> None:
    dummy_result = RunResult(pd.DataFrame(), {}, 42, {})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        return dummy_result, "runABC", None

    monkeypatch.setattr("trend.cli._ensure_dataframe", lambda _p: pd.DataFrame())
    config_obj = SimpleNamespace(
        sample_split={},
        vol_adjust={},
        portfolio={},
        run={},
        benchmarks={},
        export={},
    )
    monkeypatch.setattr(cli, "load_config", lambda path: config_obj)
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)
    recorded: dict[str, Path] = {}
    captured_report: dict[str, object] = {}

    def fake_write(out_dir: Path, *args, **kwargs) -> None:
        recorded["dir"] = out_dir

    monkeypatch.setattr("trend.cli._write_report_files", fake_write)
    monkeypatch.setattr(
        "trend.cli._resolve_returns_path",
        lambda *args, **kwargs: tmp_path / "returns.csv",
    )
    monkeypatch.setattr(
        "trend.cli.generate_unified_report",
        lambda *args, **kwargs: (
            captured_report.update(kwargs)
            or SimpleNamespace(html="<html>ok</html>", pdf_bytes=None, context={})
        ),
    )

    out_dir = tmp_path / "reports"
    exit_code = main(
        [
            "report",
            "--config",
            "config/demo.yml",
            "--out",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert recorded["dir"] == out_dir
    html_path = out_dir / "trend_report_runABC.html"
    assert html_path.read_text(encoding="utf-8") == "<html>ok</html>"
    assert captured_report["include_pdf"] is False
    assert captured_report["run_id"] == "runABC"


def test_main_report_supports_output_file_only(monkeypatch, tmp_path: Path) -> None:
    dummy_result = RunResult(pd.DataFrame(), {}, 42, {})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        return dummy_result, "xyz", None

    monkeypatch.setattr("trend.cli._ensure_dataframe", lambda _p: pd.DataFrame())
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr("trend.cli._write_report_files", lambda *a, **k: None)
    monkeypatch.setattr(
        "trend.cli._resolve_returns_path",
        lambda *args, **kwargs: tmp_path / "returns.csv",
    )
    monkeypatch.setattr(
        "trend.cli.generate_unified_report",
        lambda *a, **k: SimpleNamespace(
            html="<html>report</html>", pdf_bytes=None, context={}
        ),
    )

    target = tmp_path / "custom-report.html"
    exit_code = main(
        [
            "report",
            "--config",
            "config/demo.yml",
            "--output",
            str(target),
        ]
    )

    assert exit_code == 0
    assert target.read_text(encoding="utf-8") == "<html>report</html>"


def test_main_report_writes_pdf_when_requested(monkeypatch, tmp_path: Path) -> None:
    dummy_result = RunResult(pd.DataFrame(), {}, 42, {})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        return dummy_result, "pdf001", None

    pdf_bytes = b"%PDF-1.4\n..."
    recorded: dict[str, object] = {}

    monkeypatch.setattr("trend.cli._ensure_dataframe", lambda _p: pd.DataFrame())
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr("trend.cli._write_report_files", lambda *a, **k: None)
    monkeypatch.setattr(
        "trend.cli._resolve_returns_path",
        lambda *args, **kwargs: tmp_path / "returns.csv",
    )

    def fake_generate_unified_report(*a, **k):
        recorded.update(k)
        return SimpleNamespace(
            html="<html>with-pdf</html>", pdf_bytes=pdf_bytes, context={}
        )

    monkeypatch.setattr(
        "trend.cli.generate_unified_report",
        fake_generate_unified_report,
    )

    target = tmp_path / "report-output.html"
    exit_code = main(
        [
            "report",
            "--config",
            "config/demo.yml",
            "--output",
            str(target),
            "--pdf",
        ]
    )

    assert exit_code == 0
    assert target.read_text(encoding="utf-8") == "<html>with-pdf</html>"
    pdf_path = target.with_suffix(".pdf")
    assert pdf_path.read_bytes() == pdf_bytes
    assert recorded["include_pdf"] is True


def test_main_report_pdf_dependency_error(monkeypatch, tmp_path: Path, capsys) -> None:
    dummy_result = RunResult(pd.DataFrame(), {}, 42, {})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        return dummy_result, "pdf001", None

    monkeypatch.setattr("trend.cli._ensure_dataframe", lambda _p: pd.DataFrame())
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr("trend.cli._write_report_files", lambda *a, **k: None)
    monkeypatch.setattr(
        "trend.cli._resolve_returns_path",
        lambda *args, **kwargs: tmp_path / "returns.csv",
    )
    monkeypatch.setattr(
        "trend.cli.generate_unified_report",
        lambda *a, **k: SimpleNamespace(
            html="<html>ok</html>", pdf_bytes=None, context={}
        ),
    )

    target = tmp_path / "report-output.html"
    exit_code = main(
        [
            "report",
            "--config",
            "config/demo.yml",
            "--output",
            str(target),
            "--pdf",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "PDF generation failed" in captured.err


def test_main_report_handles_runtime_error(monkeypatch, tmp_path: Path, capsys) -> None:
    dummy_result = RunResult(pd.DataFrame(), {}, 42, {})

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        return dummy_result, "err001", None

    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("Date,Value\n2020-01-31,0.1\n", encoding="utf-8")

    monkeypatch.setattr("trend.cli._ensure_dataframe", lambda _p: pd.DataFrame())
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr("trend.cli._write_report_files", lambda *a, **k: None)
    monkeypatch.setattr(
        "trend.cli._resolve_returns_path",
        lambda *args, **kwargs: returns_path,
    )

    def raise_generate(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("trend.cli.generate_unified_report", raise_generate)

    out_dir = tmp_path / "reports"
    exit_code = main(
        [
            "report",
            "--config",
            "config/demo.yml",
            "--out",
            str(out_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "boom" in captured.err


def test_cli_report_matches_shared_generator(monkeypatch, tmp_path: Path) -> None:
    cli_result = _sample_result()
    cli_config = _sample_config()

    config_path = tmp_path / "config.yml"
    config_path.write_text("{}\n", encoding="utf-8")
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("Date,Value\n2021-01-31,0.01\n", encoding="utf-8")

    monkeypatch.setattr(
        "trend_analysis.cli._load_configuration",
        lambda path: (config_path.resolve(), cli_config),
    )
    monkeypatch.setattr(
        "trend_analysis.cli._resolve_returns_path",
        lambda *args, **kwargs: returns_path,
    )
    monkeypatch.setattr(
        "trend_analysis.cli._ensure_dataframe",
        lambda _p: pd.DataFrame({"Date": ["2021-01-31"], "Value": [0.01]}),
    )
    monkeypatch.setattr(
        "trend_analysis.cli._print_summary", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "trend_analysis.cli._write_report_files", lambda *args, **kwargs: None
    )

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        return cli_result, "cli-run", None

    monkeypatch.setattr("trend_analysis.cli._run_pipeline", fake_run_pipeline)

    report_path = tmp_path / "report.html"
    exit_code = main(
        [
            "report",
            "--config",
            str(config_path),
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    expected_artifacts = generate_unified_report(
        cli_result,
        cli_config,
        run_id="cli-run",
        include_pdf=False,
    )
    assert report_path.read_text(encoding="utf-8") == expected_artifacts.html


def test_main_report_requires_output_directory() -> None:
    exit_code = main(["report", "--config", "config/demo.yml"])
    assert exit_code == 2


def test_main_stress_passes_scenario(monkeypatch, tmp_path: Path) -> None:
    dummy_result = RunResult(pd.DataFrame(), {}, 42, {})

    captured: dict[str, object] = {}

    def fake_run_pipeline(*args, **kwargs):  # type: ignore[override]
        captured.update(kwargs)
        return dummy_result, "runXYZ", None

    monkeypatch.setattr("trend.cli._ensure_dataframe", lambda _p: pd.DataFrame())
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr("trend.cli._write_report_files", lambda *args, **kwargs: None)

    exit_code = main(
        [
            "stress",
            "--config",
            "config/demo.yml",
            "--scenario",
            "2008",
        ]
    )

    assert exit_code == 0
    assert captured["structured_log"] is False


def test_main_stress_requires_scenario() -> None:
    exit_code = main(["stress", "--config", "config/demo.yml"])
    assert exit_code == 2


def test_main_app_invokes_streamlit(monkeypatch) -> None:
    called: dict[str, int] = {}

    class DummyProcess:
        returncode = 0

    def fake_run(cmd: list[str]) -> DummyProcess:  # type: ignore[override]
        called["argc"] = len(cmd)
        return DummyProcess()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = main(["app"])

    assert exit_code == 0
    assert called["argc"] == 3


def test_prepare_export_config_updates_structure(tmp_path: Path) -> None:
    cfg = SimpleNamespace(export={"directory": "old", "formats": ["csv"]})

    cli._prepare_export_config(cfg, tmp_path, ["json", "txt"])

    assert cfg.export["directory"] == str(tmp_path)
    assert cfg.export["formats"] == ["json", "txt"]


def test_ensure_dataframe_raises_when_missing(monkeypatch, tmp_path: Path) -> None:
    def raise_missing(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise FileNotFoundError("missing.csv")

    monkeypatch.setattr(cli, "load_csv", raise_missing)

    with pytest.raises(FileNotFoundError):
        cli._ensure_dataframe(tmp_path / "missing.csv")


def test_handle_exports_with_excel(monkeypatch, tmp_path: Path) -> None:
    metrics = pd.DataFrame({"value": [1.0]})
    details = {
        "portfolio_user_weight": pd.Series([0.5], name="portfolio"),
        "benchmarks": {"bench": pd.Series([1.0], name="bench")},
        "weights_user_weight": pd.Series([0.5], name="w"),
    }
    result = RunResult(metrics, details, 1, {})

    cfg = SimpleNamespace(
        export={"directory": str(tmp_path), "formats": ["xlsx", "json"]},
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
    )

    recorded: list[tuple[str, str]] = []

    monkeypatch.setattr(
        cli.export,
        "make_summary_formatter",
        lambda *_: "formatter",
    )
    monkeypatch.setattr(
        cli.export,
        "summary_frame_from_result",
        lambda *_: pd.DataFrame({"summary": [1]}),
    )

    def fake_export_to_excel(data, path, default_sheet_formatter):  # type: ignore[override]
        recorded.append(("excel", str(path)))

    def fake_export_data(data, path, formats):  # type: ignore[override]
        recorded.append(("data", str(path)))

    monkeypatch.setattr(cli.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(cli.export, "export_data", fake_export_data)

    events: list[str] = []
    monkeypatch.setattr(
        cli,
        "_legacy_maybe_log_step",
        lambda *_args, **_kwargs: events.append("logged"),
    )

    cli._handle_exports(cfg, result, structured_log=True, run_id="run42")

    assert ("excel", str(tmp_path / "analysis.xlsx")) in recorded
    assert any(kind == "data" for kind, _ in recorded)
    assert events


def test_handle_exports_only_excel_format(monkeypatch, tmp_path: Path) -> None:
    metrics = pd.DataFrame({"value": [1.0]})
    result = RunResult(metrics, {"details": {}}, 1, {})
    cfg = SimpleNamespace(
        export={"directory": str(tmp_path), "formats": ["xlsx"]},
        sample_split={},
    )

    recorded: list[str] = []

    monkeypatch.setattr(cli.export, "make_summary_formatter", lambda *_: "fmt")
    monkeypatch.setattr(cli.export, "summary_frame_from_result", lambda *_: metrics)
    monkeypatch.setattr(
        cli.export,
        "export_to_excel",
        lambda data, path, default_sheet_formatter: recorded.append(str(path)),
    )
    monkeypatch.setattr(
        cli.export,
        "export_data",
        lambda *args, **kwargs: recorded.append("data-called"),
    )

    cli._handle_exports(cfg, result, structured_log=False, run_id="rid")

    assert recorded and all(
        entry.endswith("analysis.xlsx") for entry in recorded if entry != "data-called"
    )
    assert "data-called" not in recorded


def test_run_pipeline_sets_attributes(monkeypatch, tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        export={}, sample_split={}, run_id=None, portfolio={"transaction_cost_bps": 5.0}
    )
    returns = pd.DataFrame({"Date": ["2020-01-31"], "A": [0.1]})
    monkeypatch.chdir(tmp_path)

    metrics = pd.DataFrame({"value": [1.0]})
    details = {
        "portfolio_user_weight": {"series": [0.5]},
        "benchmarks": {"benchmark": pd.Series([0.1], name="bench")},
        "weights_user_weight": pd.Series([0.4], name="weights"),
    }
    run_result = RunResult(metrics, details, 7, {})

    monkeypatch.setattr(cli, "run_simulation", lambda *_: run_result)
    monkeypatch.setattr(
        cli.run_logging,
        "get_default_log_path",
        lambda run_id: tmp_path / f"{run_id}.json",
    )

    logs: list[str] = []
    monkeypatch.setattr(
        cli,
        "_legacy_maybe_log_step",
        lambda *_args, **_kwargs: logs.append("logged"),
    )

    handled: list[str] = []
    monkeypatch.setattr(
        cli,
        "_handle_exports",
        lambda *_args, **_kwargs: handled.append("exports"),
    )

    bundles: list[Path] = []

    def fake_write_bundle(*args, **_kwargs):  # type: ignore[override]
        bundles.append(Path(args[3]))

    monkeypatch.setattr(cli, "_write_bundle", fake_write_bundle)
    monkeypatch.setattr(
        cli.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="abcdef1234567890"),
    )

    result, run_id, log_path = cli._run_pipeline(
        cfg,
        returns,
        source_path=tmp_path / "returns.csv",
        log_file=None,
        structured_log=True,
        bundle=tmp_path / "bundle.zip",
    )

    assert isinstance(result, RunResult)
    assert run_id == "abcdef123456"
    assert log_path == tmp_path / "abcdef123456.json"
    assert hasattr(result, "portfolio")
    assert hasattr(result, "benchmark")
    assert hasattr(result, "weights")
    assert cfg.run_id == run_id
    assert logs
    assert handled
    assert bundles == [tmp_path / "bundle.zip"]


def test_run_pipeline_handles_non_dict_details(monkeypatch, tmp_path: Path) -> None:
    class FussyCfg:
        def __init__(self) -> None:
            object.__setattr__(self, "export", {})
            object.__setattr__(self, "sample_split", {})
            object.__setattr__(self, "portfolio", {"transaction_cost_bps": 1.0})

        def __setattr__(self, name: str, value: object) -> None:
            if name == "run_id":
                raise RuntimeError("cannot set run_id")
            object.__setattr__(self, name, value)

    cfg = FussyCfg()
    returns = pd.DataFrame({"Date": ["2020-01-31"], "A": [0.1]})
    run_result = RunResult(pd.DataFrame(), "summary", 5, {})
    monkeypatch.chdir(tmp_path)

    events: list[str] = []

    def record_event(*args, **kwargs) -> None:
        events.append(args[2] if len(args) >= 3 else kwargs.get("event", ""))

    monkeypatch.setattr(cli, "run_simulation", lambda *_: run_result)
    monkeypatch.setattr(cli, "_legacy_maybe_log_step", record_event)
    monkeypatch.setattr(cli, "_handle_exports", lambda *_a, **_k: None)

    result, run_id, log_path = cli._run_pipeline(
        cfg,
        returns,
        source_path=None,
        log_file=None,
        structured_log=False,
        bundle=None,
    )

    assert isinstance(result, RunResult)
    assert result.details == "summary"
    assert len(run_id) == 12
    assert log_path is None
    assert "summary_render" in events
    assert not hasattr(result, "portfolio")


def test_adjust_for_scenario_rejects_unknown() -> None:
    cfg = SimpleNamespace(sample_split={})

    with pytest.raises(TrendCLIError):
        _adjust_for_scenario(cfg, "unknown")


def test_prepare_export_config_partial_updates(tmp_path: Path) -> None:
    cfg = SimpleNamespace(export={"directory": "existing"})

    cli._prepare_export_config(cfg, None, ["csv"])

    assert cfg.export == {"directory": "existing", "formats": ["csv"]}

    cli._prepare_export_config(cfg, tmp_path, None)

    assert cfg.export == {"directory": str(tmp_path), "formats": ["csv"]}


def test_prepare_export_config_creates_export_attribute(tmp_path: Path) -> None:
    cfg = SimpleNamespace()

    cli._prepare_export_config(cfg, tmp_path, ["json"])

    assert cfg.export == {"directory": str(tmp_path), "formats": ["json"]}


def test_handle_exports_defaults_and_non_excel(monkeypatch, tmp_path: Path) -> None:
    cfg = SimpleNamespace(export={})
    metrics = pd.DataFrame({"value": [1.0]})
    result = RunResult(metrics, {"details": {}}, 1, {})

    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_DIRECTORY", str(tmp_path / "exports"))
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_FORMATS", ("csv", "json"))

    recorded: dict[str, object] = {}

    def fake_export_data(data, path, formats):  # type: ignore[override]
        recorded["data"] = data
        recorded["path"] = path
        recorded["formats"] = tuple(formats)

    monkeypatch.setattr(cli.export, "export_data", fake_export_data)
    monkeypatch.setattr(cli.export, "make_summary_formatter", lambda *_: None)

    cli._handle_exports(cfg, result, structured_log=False, run_id="abc123")

    assert recorded["formats"] == ("csv", "json")
    assert Path(recorded["path"]) == tmp_path / "exports" / "analysis"


def test_handle_exports_requires_both_directory_and_formats(monkeypatch) -> None:
    cfg = SimpleNamespace(export={"directory": "", "formats": ["csv"]})
    result = RunResult(pd.DataFrame(), {}, 1, {})

    events: list[str] = []
    monkeypatch.setattr(
        cli, "_legacy_maybe_log_step", lambda *a, **k: events.append("x")
    )

    cli._handle_exports(cfg, result, structured_log=True, run_id="rid")

    assert events == []


def test_write_bundle_appends_filename(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg = SimpleNamespace(__dict__={"a": 1})
    result = RunResult(pd.DataFrame(), {}, 1, {})

    captured: dict[str, object] = {}

    def fake_export_bundle(res, path):  # type: ignore[override]
        captured["result"] = res
        captured["path"] = path

    import trend_analysis.export.bundle

    monkeypatch.setattr(
        trend_analysis.export.bundle, "export_bundle", fake_export_bundle
    )
    monkeypatch.setattr(cli, "_legacy_maybe_log_step", lambda *a, **k: None)

    bundle_dir = tmp_path / "artifacts"
    bundle_dir.mkdir()

    cli._write_bundle(cfg, result, tmp_path / "source.csv", bundle_dir, True, "run42")

    out = capsys.readouterr().out
    expected_path = (bundle_dir / "analysis_bundle.zip").resolve()
    assert f"Bundle written: {expected_path}" in out
    assert captured["path"] == expected_path
    assert getattr(result, "input_path") == tmp_path / "source.csv"
    assert getattr(result, "config") == cfg.__dict__


def test_write_bundle_accepts_explicit_file(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg = SimpleNamespace(__dict__={"key": "value"})
    result = RunResult(pd.DataFrame(), {}, 1, {})

    recorded: dict[str, Path] = {}

    import trend_analysis.export.bundle

    monkeypatch.setattr(
        trend_analysis.export.bundle,
        "export_bundle",
        lambda res, path: recorded.update(result=res, path=path),
    )
    monkeypatch.setattr(cli, "_legacy_maybe_log_step", lambda *a, **k: None)

    bundle_file = tmp_path / "custom_bundle.zip"
    cli._write_bundle(cfg, result, None, bundle_file, False, "run00")

    out = capsys.readouterr().out
    assert f"Bundle written: {bundle_file.resolve()}" in out
    assert recorded["path"] == bundle_file.resolve()
    assert getattr(result, "config") == cfg.__dict__
    assert getattr(result, "input_path", None) is None


def test_print_summary_emits_cache_stats(monkeypatch, capsys) -> None:
    result = RunResult(pd.DataFrame(), {"details": {}}, 1, {})
    cfg = SimpleNamespace(sample_split={"in_start": "2020-01", "out_end": "2020-12"})

    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a: "Summary text")
    monkeypatch.setattr(cli, "_legacy_extract_cache_stats", lambda *_: {"hits": 3})

    cli._print_summary(cfg, result)

    out = capsys.readouterr().out
    assert "Summary text" in out
    assert "Cache statistics" in out


def test_print_summary_skips_empty_cache_stats(monkeypatch, capsys) -> None:
    result = RunResult(pd.DataFrame(), {"details": {}}, 1, {})
    cfg = SimpleNamespace(sample_split={})

    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a: "Summary text")
    monkeypatch.setattr(cli, "_legacy_extract_cache_stats", lambda *_: {})

    cli._print_summary(cfg, result)

    out = capsys.readouterr().out
    assert "Summary text" in out
    assert "Cache statistics" not in out


def test_write_report_files_creates_expected_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    metrics = pd.DataFrame({"value": [1.0]})
    result = RunResult(metrics, {"details": {}}, 1, {})
    cfg = SimpleNamespace(sample_split={})

    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a: "Report summary")

    cli._write_report_files(tmp_path, cfg, result, run_id="run7")

    metrics_path = tmp_path / "metrics_run7.csv"
    summary_path = tmp_path / "summary_run7.txt"
    details_path = tmp_path / "details_run7.json"

    assert metrics_path.exists()
    assert summary_path.read_text(encoding="utf-8") == "Report summary"
    assert json.loads(details_path.read_text(encoding="utf-8")) == {"details": {}}


def test_resolve_report_output_path_with_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    out_dir.mkdir()

    resolved = cli._resolve_report_output_path(str(out_dir), None, "run01")

    assert resolved == out_dir / "trend_report_run01.html"


def test_resolve_report_output_path_without_suffix(tmp_path: Path) -> None:
    target = tmp_path / "custom"

    resolved = cli._resolve_report_output_path(str(target), tmp_path, "run02")

    assert resolved == target / "trend_report_run02.html"


def test_json_default_handles_known_types(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1]})
    series = df["a"]
    path = tmp_path / "file.txt"

    assert cli._json_default(df)["a"][0] == 1
    assert cli._json_default(series)[0] == 1
    assert cli._json_default(path) == str(path)

    with pytest.raises(TypeError):
        cli._json_default(123)


def test_main_handles_file_not_found(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_load_configuration(path: str):  # type: ignore[override]
        raise FileNotFoundError(path)

    monkeypatch.setattr(cli, "_load_configuration", fake_load_configuration)

    exit_code = cli.main(["run", "--config", str(tmp_path / "missing.yml")])

    assert exit_code == 2
    assert "Error:" in capsys.readouterr().err


def test_main_reports_unknown_command(monkeypatch, capsys) -> None:
    parser = SimpleNamespace(
        parse_args=lambda _argv: SimpleNamespace(subcommand="mystery", config="cfg.yml")
    )

    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    monkeypatch.setattr(
        cli,
        "_load_configuration",
        lambda path: (Path(path), SimpleNamespace(data={"csv_path": "returns.csv"})),
    )
    monkeypatch.setattr(
        cli, "_resolve_returns_path", lambda *_args: Path("returns.csv")
    )
    monkeypatch.setattr(cli, "_ensure_dataframe", lambda *_args: pd.DataFrame())
    monkeypatch.setattr(cli, "_determine_seed", lambda *_args: 0)

    exit_code = cli.main(["mystery", "--config", "cfg.yml"])

    assert exit_code == 2
    assert "Unknown command" in capsys.readouterr().err
