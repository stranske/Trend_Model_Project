"""Focused tests for ``trend.cli`` helper behaviour and command dispatch."""

from __future__ import annotations

import json
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend import cli as trend_cli
from trend_analysis.logging_setup import RUNS_ROOT


class DummyResult:
    def __init__(self) -> None:
        self.details = {
            "portfolio_equal_weight": [1, 2, 3],
            "benchmarks": {"ref": [0.1, 0.2]},
            "weights_user_weight": [0.5, 0.5],
        }
        turnover_idx = pd.date_range("2020-01-31", periods=2, freq="ME")
        self.details["risk_diagnostics"] = {
            "turnover": pd.Series([0.1, 0.2], index=turnover_idx),
            "turnover_value": 0.3,
        }
        self.metrics = pd.DataFrame({"metric": [1]})


def _make_config(**kwargs: object) -> types.SimpleNamespace:
    base = {
        "data": {"csv_path": "returns.csv"},
        "export": {"directory": "out", "formats": ["csv"]},
        "sample_split": {
            "in_start": "2020-01",
            "in_end": "2020-12",
            "out_start": "2021-01",
            "out_end": "2021-12",
        },
        "portfolio": {"transaction_cost_bps": 10.0},
    }
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def test_noop_maybe_log_step_returns_none() -> None:
    assert trend_cli._noop_maybe_log_step(True, "id", "event", "msg") is None


def test_resolve_returns_path_relative(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("config", encoding="utf-8")
    config = _make_config()
    resolved = trend_cli._resolve_returns_path(cfg_path, config, None)
    assert resolved == (cfg_path.parent / "returns.csv").resolve()

    override = trend_cli._resolve_returns_path(
        cfg_path, config, str(tmp_path / "override.csv")
    )
    assert override == (tmp_path / "override.csv").resolve()


def test_resolve_returns_path_requires_csv(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("config", encoding="utf-8")
    with pytest.raises(trend_cli.TrendCLIError):
        trend_cli._resolve_returns_path(cfg_path, types.SimpleNamespace(data={}), None)


def test_ensure_dataframe_validates_load(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(
        trend_cli, "load_csv", lambda path: frame if "ok" in path else None
    )
    assert trend_cli._ensure_dataframe(Path("ok.csv")).equals(frame)

    with pytest.raises(FileNotFoundError):
        trend_cli._ensure_dataframe(Path("missing.csv"))


def test_determine_seed_prefers_override_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(seed=99)
    assert trend_cli._determine_seed(config, 7) == 7
    assert config.seed == 7

    monkeypatch.setenv("TREND_SEED", "42")
    assert trend_cli._determine_seed(config, None) == 42

    monkeypatch.setenv("TREND_SEED", "invalid")
    fallback_cfg = _make_config(seed=7)
    assert trend_cli._determine_seed(fallback_cfg, None) == 7


def test_determine_seed_handles_setattr_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Frozen:
        def __init__(self) -> None:
            object.__setattr__(self, "seed", 11)

        def __setattr__(self, name: str, value: object) -> None:
            raise RuntimeError

    monkeypatch.delenv("TREND_SEED", raising=False)
    cfg = Frozen()
    assert trend_cli._determine_seed(cfg, None) == 11


def test_prepare_export_config_merges_directory_and_formats() -> None:
    config = _make_config(export={"formats": ["json"]})
    trend_cli._prepare_export_config(config, Path("custom"), ["csv", "JSON"])
    assert config.export == {"directory": "custom", "formats": ["csv", "JSON"]}

    trend_cli._prepare_export_config(config, None, None)
    assert config.export == {"directory": "custom", "formats": ["csv", "JSON"]}


def test_prepare_export_config_ignores_setattr_failures() -> None:
    class Guarded(types.SimpleNamespace):
        def __setattr__(self, name: str, value: object) -> None:
            if name == "export" and hasattr(self, "export"):
                raise RuntimeError
            super().__setattr__(name, value)

    cfg = Guarded(export={})
    trend_cli._prepare_export_config(cfg, Path("dir"), ["txt"])


def test_handle_exports_invokes_exporters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config(export={"directory": str(tmp_path), "formats": ["xlsx", "csv"]})
    result = DummyResult()

    summary_called: list[dict[str, object]] = []
    export_calls: list[tuple] = []

    monkeypatch.setattr(
        trend_cli.export, "make_summary_formatter", lambda *_: lambda name, df: df
    )
    monkeypatch.setattr(
        trend_cli.export, "summary_frame_from_result", lambda *_: pd.DataFrame()
    )

    def fake_export_to_excel(data, path, default_sheet_formatter=None):
        summary_called.append({"path": path, "data": data})
        Path(path).touch()

    monkeypatch.setattr(trend_cli.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(
        trend_cli.export,
        "export_data",
        lambda data, path, formats: export_calls.append((tuple(sorted(formats)), path)),
    )
    monkeypatch.setattr(
        trend_cli, "_legacy_maybe_log_step", lambda *args, **kwargs: None
    )

    trend_cli._handle_exports(cfg, result, structured_log=True, run_id="abc")

    assert summary_called and export_calls
    assert (tmp_path / "analysis.xlsx").exists()


def test_handle_exports_without_excel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config(export={"directory": str(tmp_path), "formats": ["json"]})
    result = DummyResult()

    calls: list[tuple] = []
    monkeypatch.setattr(
        trend_cli.export,
        "export_data",
        lambda data, path, formats: calls.append((tuple(formats), path)),
    )
    monkeypatch.setattr(
        trend_cli, "_legacy_maybe_log_step", lambda *args, **kwargs: None
    )

    trend_cli._handle_exports(cfg, result, structured_log=False, run_id="abc")
    assert calls == [(("json",), str(Path(cfg.export["directory"]) / "analysis"))]


def test_run_pipeline_sets_metadata_and_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config()
    returns = pd.DataFrame({"x": [1, 2, 3]})
    result = DummyResult()
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(trend_cli, "run_simulation", lambda *_: result)
    monkeypatch.setattr(
        trend_cli, "_legacy_maybe_log_step", lambda *args, **kwargs: None
    )
    handled: list[tuple] = []
    monkeypatch.setattr(
        trend_cli, "_handle_exports", lambda *args, **kwargs: handled.append(args)
    )
    written: list[tuple] = []
    monkeypatch.setattr(
        trend_cli, "_write_bundle", lambda *args, **kwargs: written.append(args)
    )
    monkeypatch.setattr(
        trend_cli.run_logging,
        "get_default_log_path",
        lambda run_id: Path(tmp_path / f"{run_id}.log"),
    )

    result_obj, run_id, log_path = trend_cli._run_pipeline(
        cfg,
        returns,
        source_path=tmp_path / "returns.csv",
        log_file=None,
        structured_log=True,
        bundle=tmp_path / "bundle.zip",
    )

    assert isinstance(run_id, str) and len(run_id) == 12
    assert log_path == tmp_path / f"{run_id}.log"
    assert handled and written
    assert result_obj is result
    ledger = Path("perf") / run_id / "turnover.csv"
    assert ledger.exists()
    df = pd.read_csv(ledger)
    assert df["turnover"].sum() == pytest.approx(0.3)
    log_path = trend_cli._LAST_PERF_LOG_PATH
    assert log_path is not None
    assert log_path.name == "app.log"
    assert log_path.exists()
    assert RUNS_ROOT in log_path.parents


def test_run_pipeline_requires_transaction_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config()
    cfg.portfolio = {}
    returns = pd.DataFrame({"x": [1, 2, 3]})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trend_cli, "run_simulation", lambda *_: DummyResult())
    monkeypatch.setattr(
        trend_cli, "_legacy_maybe_log_step", lambda *args, **kwargs: None
    )

    with pytest.raises(trend_cli.TrendCLIError, match="transaction_cost_bps"):
        trend_cli._run_pipeline(
            cfg,
            returns,
            source_path=None,
            log_file=None,
            structured_log=False,
            bundle=None,
        )


def test_write_bundle_normalises_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    recorded: list[Path] = []
    monkeypatch.setattr(
        "trend_analysis.export.bundle.export_bundle",
        lambda result, path: recorded.append(path),
    )
    monkeypatch.setattr(
        trend_cli, "_legacy_maybe_log_step", lambda *args, **kwargs: None
    )

    result = DummyResult()
    cfg = _make_config()
    trend_cli._write_bundle(
        cfg,
        result,
        tmp_path / "returns.csv",
        bundle_dir,
        structured_log=False,
        run_id="abc",
    )

    assert recorded[0].name == "analysis_bundle.zip"
    assert getattr(result, "input_path") == tmp_path / "returns.csv"


def test_print_summary_emits_cache_stats(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = DummyResult()
    cfg = _make_config(sample_split={"in_start": "2020-01", "in_end": "2020-02"})
    monkeypatch.setattr(trend_cli.export, "format_summary_text", lambda *_: "SUMMARY")
    monkeypatch.setattr(
        trend_cli, "_legacy_extract_cache_stats", lambda *_: {"hits": 2}
    )

    trend_cli._print_summary(cfg, result)
    out = capsys.readouterr().out
    assert "SUMMARY" in out and "Cache statistics" in out


def test_write_report_files_creates_expected_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = DummyResult()
    cfg = _make_config(sample_split={"in_start": "2020-01", "in_end": "2020-02"})
    monkeypatch.setattr(trend_cli.export, "format_summary_text", lambda *_: "SUMMARY")

    trend_cli._write_report_files(tmp_path, cfg, result, run_id="xyz")

    metrics_path = tmp_path / "metrics_xyz.csv"
    summary_path = tmp_path / "summary_xyz.txt"
    details_path = tmp_path / "details_xyz.json"
    turnover_path = tmp_path / "turnover.csv"
    assert metrics_path.exists() and summary_path.exists() and details_path.exists()
    assert turnover_path.exists()
    data = json.loads(details_path.read_text())
    assert "benchmarks" in data
    turnover_df = pd.read_csv(turnover_path)
    assert turnover_df["turnover"].sum() == pytest.approx(0.3)


def test_adjust_for_scenario_updates_config() -> None:
    cfg = _make_config(sample_split={})
    trend_cli._adjust_for_scenario(cfg, "2008")
    assert cfg.sample_split["out_end"] == "2009-12"

    with pytest.raises(trend_cli.TrendCLIError):
        trend_cli._adjust_for_scenario(cfg, "missing")


def test_adjust_for_scenario_handles_attr_failure() -> None:
    class Guarded:
        def __init__(self) -> None:
            self.sample_split = {}

        def __setattr__(self, name: str, value: object) -> None:
            if name == "sample_split" and hasattr(self, "sample_split"):
                raise RuntimeError
            super().__setattr__(name, value)

    cfg = Guarded()
    trend_cli._adjust_for_scenario(cfg, "2008")


def test_load_configuration_reads_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("config", encoding="utf-8")

    monkeypatch.setattr(trend_cli, "load_config", lambda path: {"version": "1"})

    path, cfg = trend_cli._load_configuration(str(cfg_file))
    assert path == cfg_file.resolve()
    assert cfg == {"version": "1"}


def test_load_configuration_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        trend_cli._load_configuration(str(tmp_path / "absent.yml"))


def test_main_run_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _make_config()
    cfg_path = tmp_path / "cfg.yml"
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("csv", encoding="utf-8")

    monkeypatch.setattr(
        trend_cli, "_load_configuration", lambda path: (Path(path), cfg)
    )
    monkeypatch.setattr(
        trend_cli, "_ensure_dataframe", lambda path: pd.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(trend_cli, "_determine_seed", lambda cfg, override: 123)
    monkeypatch.setattr(
        trend_cli,
        "_run_pipeline",
        lambda *_args, **_kwargs: (DummyResult(), "run123", tmp_path / "log.jsonl"),
    )
    monkeypatch.setattr(trend_cli, "_print_summary", lambda *args, **kwargs: None)

    exit_code = trend_cli.main(
        [
            "run",
            "--config",
            str(cfg_path),
            "--returns",
            str(returns_path),
            "--log-file",
            str(tmp_path / "custom.log"),
        ]
    )

    assert exit_code == 0
    assert "Structured log" in capsys.readouterr().out


def test_main_report_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _make_config()
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("csv", encoding="utf-8")

    monkeypatch.setattr(
        trend_cli, "_load_configuration", lambda path: (Path(path), cfg)
    )
    monkeypatch.setattr(
        trend_cli, "_ensure_dataframe", lambda path: pd.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(trend_cli, "_determine_seed", lambda cfg, override: 123)
    monkeypatch.setattr(
        trend_cli,
        "_run_pipeline",
        lambda *_args, **_kwargs: (DummyResult(), "run123", None),
    )
    monkeypatch.setattr(trend_cli, "_print_summary", lambda *args, **kwargs: None)
    created: list[Path] = []
    monkeypatch.setattr(
        trend_cli,
        "_write_report_files",
        lambda out, cfg, result, run_id: created.append(out),
    )
    monkeypatch.setattr(
        trend_cli,
        "generate_unified_report",
        lambda *a, **k: SimpleNamespace(
            html="<html>report</html>", pdf_bytes=None, context={}
        ),
    )

    exit_code = trend_cli.main(
        [
            "report",
            "--config",
            str(tmp_path / "cfg.yml"),
            "--returns",
            str(returns_path),
            "--out",
            str(tmp_path / "reports"),
            "--formats",
            "csv",
        ]
    )

    assert exit_code == 0 and created


def test_main_stress_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _make_config()
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("csv", encoding="utf-8")

    monkeypatch.setattr(
        trend_cli, "_load_configuration", lambda path: (Path(path), cfg)
    )
    monkeypatch.setattr(
        trend_cli, "_ensure_dataframe", lambda path: pd.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(trend_cli, "_determine_seed", lambda cfg, override: 123)
    monkeypatch.setattr(
        trend_cli,
        "_run_pipeline",
        lambda *_args, **_kwargs: (DummyResult(), "run123", None),
    )
    monkeypatch.setattr(trend_cli, "_print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(trend_cli, "_write_report_files", lambda *args, **kwargs: None)

    exit_code = trend_cli.main(
        [
            "stress",
            "--config",
            str(tmp_path / "cfg.yml"),
            "--returns",
            str(returns_path),
            "--scenario",
            "2008",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0 and "Stress scenario" in captured


def test_main_stress_with_export_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _make_config()
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("csv", encoding="utf-8")

    monkeypatch.setattr(
        trend_cli, "_load_configuration", lambda path: (Path(path), cfg)
    )
    monkeypatch.setattr(
        trend_cli, "_ensure_dataframe", lambda path: pd.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(trend_cli, "_determine_seed", lambda cfg, override: 123)
    monkeypatch.setattr(
        trend_cli,
        "_run_pipeline",
        lambda *_args, **_kwargs: (DummyResult(), "run123", None),
    )
    monkeypatch.setattr(trend_cli, "_print_summary", lambda *args, **kwargs: None)

    wrote: list[Path] = []
    monkeypatch.setattr(
        trend_cli,
        "_write_report_files",
        lambda out, cfg, result, run_id: wrote.append(out),
    )

    exit_code = trend_cli.main(
        [
            "stress",
            "--config",
            str(tmp_path / "cfg.yml"),
            "--returns",
            str(returns_path),
            "--scenario",
            "2008",
            "--out",
            str(tmp_path / "stress"),
        ]
    )

    assert exit_code == 0 and wrote


def test_main_app_command(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = types.SimpleNamespace(returncode=5)
    monkeypatch.setattr(trend_cli.subprocess, "run", lambda args: proc)
    assert trend_cli.main(["app"]) == 5


def test_main_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trend_cli, "_load_configuration", lambda *_: (Path("cfg.yml"), _make_config())
    )
    exit_code = trend_cli.main(["run"])
    assert exit_code == 2

    monkeypatch.setattr(
        trend_cli, "_load_configuration", lambda *_: (Path("cfg.yml"), _make_config())
    )
    monkeypatch.setattr(
        trend_cli,
        "_ensure_dataframe",
        lambda *_: exec('raise FileNotFoundError("missing")'),
    )
    exit_code = trend_cli.main(["run", "--config", "cfg.yml"])
    assert exit_code == 2


def test_main_unknown_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trend_cli, "_load_configuration", lambda *_: (Path("cfg.yml"), _make_config())
    )
    monkeypatch.setattr(
        trend_cli, "_ensure_dataframe", lambda *_: pd.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(trend_cli, "_determine_seed", lambda *_: 1)
    with pytest.raises(SystemExit) as excinfo:
        trend_cli.main(["unknown", "--config", "cfg.yml", "--returns", "data.csv"])
    assert excinfo.value.code == 2
