"""Focused coverage for ``trend_analysis.cli`` entrypoints and helpers."""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pandas as pd
import numpy as np
import pytest

from trend_analysis import cli
from trend_analysis.io.market_data import MarketDataValidationError


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        sample_split={
            "in_start": "2020-01",
            "in_end": "2020-03",
            "out_start": "2020-04",
            "out_end": "2020-05",
        },
        export={"directory": None, "formats": None, "filename": "analysis"},
        vol_adjust={},
        portfolio={},
        benchmarks={},
        metrics={},
        run={},
    )


@pytest.mark.parametrize("preset_missing", ["spec", "portfolio"])
def test_main_run_reports_unknown_presets(monkeypatch, preset_missing: str, capsys):
    cfg = _make_config()
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: None)
    monkeypatch.setattr(cli, "load_market_data_csv", lambda path: None)

    if preset_missing == "spec":
        monkeypatch.setattr(cli, "get_trend_spec_preset", lambda name: (_ for _ in ()).throw(KeyError(name)))
        monkeypatch.setattr(cli, "list_trend_spec_presets", lambda: ["alpha", "beta"])
    else:
        monkeypatch.setattr(cli, "get_trend_spec_preset", lambda name: SimpleNamespace(name=name))
        monkeypatch.setattr(cli, "_apply_trend_spec_preset", lambda *_: None)
        monkeypatch.setattr(cli, "get_trend_preset", lambda name: (_ for _ in ()).throw(KeyError(name)))
        monkeypatch.setattr(cli, "list_preset_slugs", lambda: ["balanced", "aggressive"])

    rc = cli.main([
        "run",
        "--config",
        "cfg.yml",
        "--input",
        "returns.csv",
        "--preset",
        "missing",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "Unknown preset" in captured.err


def test_main_run_handles_market_data_validation_error(monkeypatch, capsys):
    cfg = _make_config()
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: None)
    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: (_ for _ in ()).throw(MarketDataValidationError("bad data")),
    )

    rc = cli.main([
        "run",
        "--config",
        "cfg.yml",
        "--input",
        "returns.csv",
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "bad data" in captured.err


def test_main_check_flag_short_circuits(monkeypatch):
    called: dict[str, int] = {}

    def fake_check(lock_path=None):
        called["count"] = called.get("count", 0) + 1
        return 0

    monkeypatch.setattr(cli, "check_environment", fake_check)

    assert cli.main(["--check"]) == 0
    assert called == {"count": 1}


def test_main_defaults_to_sys_argv(monkeypatch):
    calls: list[None] = []

    def fake_check():
        calls.append(None)
        return 0

    monkeypatch.setattr(cli, "check_environment", fake_check)
    monkeypatch.setattr(sys, "argv", ["trend-model", "--check"])

    assert cli.main() == 0
    assert calls == [None]


def test_main_gui_invokes_streamlit(monkeypatch):
    recorded: dict[str, object] = {}

    class Proc:
        returncode = 5

    def fake_run(args):
        recorded["args"] = args
        return Proc()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    rc = cli.main(["gui"])

    assert rc == 5
    assert recorded["args"] == ["streamlit", "run", str(cli.APP_PATH)]


def test_main_handles_check_flag_with_subcommand(monkeypatch):
    calls: list[None] = []

    def fake_check():
        calls.append(None)
        return 0

    def fake_parse(self, argv=None):
        return SimpleNamespace(check=True)

    monkeypatch.setattr(cli, "check_environment", fake_check)
    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", fake_parse)

    rc = cli.main([])
    assert rc == 0
    assert calls == [None]


def test_main_run_success_path_covers_exports_and_bundle(monkeypatch, tmp_path, capsys):
    cfg = _make_config()
    applied: dict[str, SimpleNamespace | None] = {"spec": None, "portfolio": None}
    toggles: list[bool] = []
    logged: list[str] = []
    excel_calls: list[tuple[dict[str, pd.DataFrame], str]] = []
    data_calls: list[tuple[dict[str, pd.DataFrame], str, tuple[str, ...]]] = []
    bundle_calls: list[Path] = []

    def fake_apply(cfg_obj, preset):
        applied["spec"] = preset

    def fake_apply_portfolio(cfg_obj, preset):
        applied["portfolio"] = preset

    def fake_log_step(run_id: str, event: str, message: str, **_fields):
        logged.append(event)

    def fake_export_to_excel(payload, dest, **_):
        excel_calls.append((payload, dest))

    def fake_export_data(payload, dest, formats):
        data_calls.append((payload, dest, tuple(sorted(formats))))

    def fake_export_bundle(result, path):
        bundle_calls.append(path)

    monkeypatch.setenv("TREND_SEED", "7")
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "_apply_trend_spec_preset", fake_apply)
    monkeypatch.setattr(cli, "get_trend_spec_preset", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(cli, "get_trend_preset", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(cli, "apply_trend_preset", fake_apply_portfolio)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: toggles.append(enabled))
    frame = pd.DataFrame({
        "Date": pd.date_range("2020-01-31", periods=3, freq="ME"),
        "Fund": [0.01, 0.02, 0.03],
    })
    monkeypatch.setattr(cli, "load_market_data_csv", lambda path: SimpleNamespace(frame=frame))
    run_result = SimpleNamespace(
        metrics=pd.DataFrame({"metric": [1.0]}),
        details={
            "snapshots": [
                {"entries": 1, "hits": 2, "misses": 0, "incremental_updates": 1},
                [
                    np.array([1, 2, 3]),
                    {
                        "entries": 4.0,
                        "hits": 5.0,
                        "misses": 6.0,
                        "incremental_updates": 7.0,
                    },
                ],
            ],
            "portfolio_user_weight": pd.Series([0.2, 0.8], name="portfolio"),
            "benchmarks": {"bm": pd.Series([0.1, 0.2], name="bm")},
            "weights_user_weight": pd.Series([0.5, 0.5], name="weights"),
        },
        seed=123,
        environment={"python": "3.11"},
    )
    monkeypatch.setattr(cli, "run_simulation", lambda *_, **__: run_result)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "SUMMARY")
    monkeypatch.setattr(cli.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(cli.export, "export_data", fake_export_data)
    monkeypatch.setattr(cli.run_logging, "get_default_log_path", lambda run_id: tmp_path / f"{run_id}.jsonl")
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda *_, **__: None)
    monkeypatch.setattr(cli, "_log_step", fake_log_step)
    monkeypatch.setattr("trend_analysis.export.bundle.export_bundle", fake_export_bundle)

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    rc = cli.main([
        "run",
        "--config",
        "cfg.yml",
        "--input",
        "returns.csv",
        "--seed",
        "123",
        "--preset",
        "balanced",
        "--bundle",
        str(bundle_dir),
    ])

    out = capsys.readouterr()

    assert rc == 0
    assert cfg.seed == 123  # CLI flag takes precedence over env var
    assert toggles == [True]
    assert applied["spec"].name == "balanced"
    assert applied["portfolio"].name == "balanced"
    assert any("SUMMARY" in stream for stream in (out.out, out.err))
    assert "Cache statistics" in out.out
    assert excel_calls and excel_calls[0][1].endswith("analysis.xlsx")
    assert data_calls and set(data_calls[0][2])
    assert bundle_calls == [bundle_dir / "analysis_bundle.zip"]
    assert {"start", "export_start", "export_complete", "bundle_complete", "end"}.issubset(set(logged))


def test_main_prefers_env_seed_and_handles_run_id_failure(monkeypatch, tmp_path, capsys):
    base_cfg = _make_config()

    class ImmutableCfg(SimpleNamespace):
        def __setattr__(self, name: str, value: object) -> None:
            if name == "run_id":
                raise RuntimeError("immutable")
            super().__setattr__(name, value)

    cfg = ImmutableCfg(**base_cfg.__dict__)
    cfg.export = {"directory": None, "formats": None, "filename": "analysis"}
    toggles: list[bool] = []
    monkeypatch.setenv("TREND_SEED", "17")
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: toggles.append(enabled))
    monkeypatch.setattr(cli, "load_market_data_csv", lambda path: SimpleNamespace(frame=pd.DataFrame({"v": [1.0]})))
    run_result = SimpleNamespace(
        metrics=pd.DataFrame({"metric": [1.0]}),
        details={
            "portfolio_user_weight": [0.5, 0.5],
            "portfolio_equal_weight": [0.4, 0.6],
            "benchmarks": {"bench": [0.1, 0.2]},
            "weights_user_weight": [0.3, 0.7],
        },
        seed=11,
        environment={},
    )
    monkeypatch.setattr(cli, "run_simulation", lambda *_, **__: run_result)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "TEXT")
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *a, **k: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_logging, "get_default_log_path", lambda run_id: tmp_path / f"{run_id}.jsonl")
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_logging, "log_step", lambda *a, **k: None)

    rc = cli.main([
        "run",
        "--config",
        "cfg.yml",
        "--input",
        "returns.csv",
    ])

    out = capsys.readouterr().out

    assert rc == 0
    assert cfg.seed == 17
    assert toggles == [True]
    assert "TEXT" in out
    assert hasattr(run_result, "portfolio")
    assert hasattr(run_result, "benchmark")
    assert hasattr(run_result, "weights")


def test_main_run_handles_custom_formats_and_no_structured_log(monkeypatch, tmp_path, capsys):
    cfg = _make_config()
    cfg.export = {"directory": str(tmp_path / "out"), "formats": ["excel", "json"], "filename": "report"}
    toggles: list[bool] = []
    log_calls: list[str] = []
    excel_targets: list[str] = []
    data_targets: list[tuple[str, tuple[str, ...]]] = []
    bundle_targets: list[Path] = []

    out_dir = Path(cfg.export["directory"])
    out_dir.mkdir()

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: toggles.append(enabled))
    monkeypatch.setattr(cli, "load_market_data_csv", lambda path: pd.DataFrame({"v": [1.0]}))
    run_result = SimpleNamespace(
        metrics=pd.DataFrame({"metric": [2.0]}),
        details={"cache": {"entries": 2, "hits": 3, "misses": 1, "incremental_updates": 4}},
        seed=11,
        environment={},
    )
    monkeypatch.setattr(cli, "run_simulation", lambda *_, **__: run_result)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "ALT SUMMARY")
    monkeypatch.setattr(
        cli.export,
        "export_to_excel",
        lambda payload, dest, **_k: excel_targets.append(dest),
    )
    monkeypatch.setattr(
        cli.export,
        "export_data",
        lambda _payload, dest, formats: data_targets.append((dest, tuple(sorted(formats)))),
    )
    monkeypatch.setattr(cli.run_logging, "get_default_log_path", lambda run_id: tmp_path / f"alt-{run_id}.jsonl")
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda *_, **__: log_calls.append("init"))
    monkeypatch.setattr(cli, "_log_step", lambda *_a, **_k: log_calls.append("log"))
    monkeypatch.setattr("trend_analysis.export.bundle.export_bundle", lambda rr, dest: bundle_targets.append(dest))

    rc = cli.main(
        [
            "run",
            "--config",
            "cfg.yml",
            "--input",
            "returns.csv",
            "--no-structured-log",
            "--no-cache",
            "--log-file",
            str(tmp_path / "custom.jsonl"),
            "--bundle",
        ]
    )

    out = capsys.readouterr()

    assert rc == 0
    assert toggles == [False]
    assert "ALT SUMMARY" in out.out
    assert excel_targets[0].endswith("report.xlsx")
    assert data_targets[0][1] == ("json",)
    assert bundle_targets == [Path("analysis_bundle.zip")]
    assert log_calls == []  # logging disabled when --no-structured-log is set


def test_main_returns_zero_when_no_results(monkeypatch, tmp_path, capsys):
    cfg = _make_config()
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: None)
    monkeypatch.setattr(cli, "load_market_data_csv", lambda path: SimpleNamespace(frame=pd.DataFrame({"v": [1.0]})))
    run_result = SimpleNamespace(metrics=pd.DataFrame(), details={}, seed=5, environment={})
    monkeypatch.setattr(cli, "run_simulation", lambda *_, **__: run_result)
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "IGNORED")
    monkeypatch.setattr(cli.run_logging, "get_default_log_path", lambda run_id: tmp_path / f"{run_id}.jsonl")
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_logging, "log_step", lambda *a, **k: None)

    rc = cli.main([
        "run",
        "--config",
        "cfg.yml",
        "--input",
        "returns.csv",
    ])

    out = capsys.readouterr().out

    assert rc == 0
    assert "No results" in out


def test_main_legacy_path_builds_bundle_shim(monkeypatch, tmp_path, capsys):
    cfg = _make_config()
    cfg.sample_split = {}
    cfg.export = {"directory": str(tmp_path / "legacy_out"), "formats": ["excel"], "filename": "legacy"}
    Path(cfg.export["directory"]).mkdir()

    class DummyRR:
        def __init__(self, metrics, details, seed, env):
            self.metrics = metrics
            self.details = details
            self.seed = seed
            self.environment = env

    bundle_targets: list[Path] = []

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: None)
    monkeypatch.setattr(cli, "load_market_data_csv", lambda path: SimpleNamespace(frame=pd.DataFrame({"v": [1.0]})))
    monkeypatch.setattr(cli.pipeline, "run", lambda cfg: pd.DataFrame({"metric": [1.0]}))
    monkeypatch.setattr(cli.pipeline, "run_full", lambda cfg: {"portfolio": [1, 2, 3]})
    monkeypatch.setattr(cli.export, "format_summary_text", lambda *a, **k: "LEGACY")
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *a, **k: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_logging, "get_default_log_path", lambda run_id: tmp_path / f"legacy-{run_id}.jsonl")
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda *a, **k: None)
    monkeypatch.setattr(cli.run_logging, "log_step", lambda *a, **k: None)
    monkeypatch.setattr("trend_analysis.api.RunResult", DummyRR)
    monkeypatch.setattr(
        "trend_analysis.export.bundle.export_bundle",
        lambda rr, dest: bundle_targets.append(dest),
    )

    rc = cli.main([
        "run",
        "--config",
        "cfg.yml",
        "--input",
        "returns.csv",
        "--bundle",
        str(tmp_path / "legacy_bundle.zip"),
    ])

    out = capsys.readouterr().out

    assert rc == 0
    assert "LEGACY" in out
    assert bundle_targets == [tmp_path / "legacy_bundle.zip"]


def test_load_market_data_csv_applies_defaults(monkeypatch):
    recorded: dict[str, object] = {}

    def fake_load_csv(path: str, **kwargs: object):
        recorded.update(kwargs)
        return "loaded"

    monkeypatch.setattr(cli, "load_csv", fake_load_csv)

    result = cli.load_market_data_csv("data.csv")

    assert result == "loaded"
    assert recorded["errors"] == "raise"
    assert recorded["include_date_column"] is True


def test_apply_trend_spec_preset_handles_mapping_and_frozen(monkeypatch):
    preset_payload = {"window": 5, "lag": 2}

    @dataclass
    class DummyPreset:
        name: str = "Test"

        def as_signal_config(self) -> dict[str, int]:
            return dict(preset_payload)

    preset = DummyPreset()

    cfg_dict: dict[str, object] = {"signals": {"existing": 1}}
    cli._apply_trend_spec_preset(cfg_dict, preset)
    assert cfg_dict["signals"]["window"] == 5
    assert cfg_dict["trend_spec_preset"] == "Test"

    class FrozenCfg:
        def __init__(self) -> None:
            object.__setattr__(self, "signals", {})
            object.__setattr__(self, "trend_spec_preset", None)

        def __setattr__(self, name: str, value: object) -> None:
            raise ValueError("frozen")

    frozen = FrozenCfg()
    cli._apply_trend_spec_preset(frozen, preset)
    assert frozen.signals["lag"] == 2
    assert frozen.trend_spec_preset == "Test"

    class OddCfg:
        def __init__(self) -> None:
            self.signals = 0
            self.trend_spec_preset = None

    odd = OddCfg()
    cli._apply_trend_spec_preset(odd, preset)
    assert isinstance(odd.signals, dict)
    assert odd.signals["window"] == 5


def test_check_environment_variants(monkeypatch, tmp_path, capsys):
    lock = tmp_path / "req.lock"
    lock.write_text("pkg-a==1.0\npkg-b==2.0\n# comment\ninvalid>=1\n")

    versions = {"pkg-a": "1.0", "pkg-b": "2.0"}

    def fake_version(name: str) -> str:
        if name not in versions:
            raise metadata.PackageNotFoundError
        return versions[name]

    monkeypatch.setattr(cli.metadata, "version", fake_version)

    ok_rc = cli.check_environment(lock)
    out_ok = capsys.readouterr().out
    assert ok_rc == 0
    assert "pkg-a 1.0" in out_ok
    assert "pkg-b 2.0" in out_ok

    versions["pkg-b"] = "0.9"
    fail_rc = cli.check_environment(lock)
    out_fail = capsys.readouterr().out
    assert fail_rc == 1
    assert "Mismatches detected" in out_fail
    assert "pkg-b" in out_fail

    del versions["pkg-b"]
    missing_rc = cli.check_environment(lock)
    out_missing = capsys.readouterr().out
    assert missing_rc == 1
    assert "pkg-b not installed" in out_missing


def test_check_environment_missing_lock_file(tmp_path, capsys):
    missing = tmp_path / "absent.lock"
    rc = cli.check_environment(missing)
    out = capsys.readouterr().out
    assert rc == 1
    assert f"Lock file not found: {missing}" in out


def test_extract_cache_stats_prefers_latest(monkeypatch):
    payload = {
        "first": {"entries": 1, "hits": 1, "misses": 1, "incremental_updates": 1},
        "second": [
            "ignored",
            {
                "entries": 3.0,
                "hits": 4.0,
                "misses": 5.0,
                "incremental_updates": 6.0,
            },
        ],
    }

    stats = cli._extract_cache_stats(payload)

    assert stats == {"entries": 3, "hits": 4, "misses": 5, "incremental_updates": 6}


def test_extract_cache_stats_ignores_non_integer_values():
    payload = {"bad": {"entries": "x", "hits": 1, "misses": 2, "incremental_updates": 3}}
    assert cli._extract_cache_stats(payload) is None


def test_log_step_delegates_to_run_logging(monkeypatch):
    recorded: list[tuple[str, str, str, dict[str, object]]] = []

    def fake_log_step(run_id: str, event: str, message: str, **fields: object) -> None:
        recorded.append((run_id, event, message, fields))

    monkeypatch.setattr(cli.run_logging, "log_step", fake_log_step)

    cli._log_step("abc", "event", "message", level="DEBUG", extra=True)

    assert recorded == [("abc", "event", "message", {"level": "DEBUG", "extra": True})]


def test_maybe_log_step_only_logs_when_enabled(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cli,
        "_log_step",
        lambda run_id, event, message, **_k: calls.append((event, message)),
    )

    cli.maybe_log_step(False, "run", "event", "message")
    cli.maybe_log_step(True, "run", "event", "message")

    assert calls == [("event", "message")]


def test_cli_compatibility_wrappers_delegate(monkeypatch):
    stub = ModuleType("trend.cli")
    stub_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    stub._load_configuration = lambda path: (Path(path), {"loaded": True})  # type: ignore[attr-defined]

    def record(name):
        def _inner(*args, **kwargs):
            stub_calls.append((name, args, kwargs))
            if name == "run_pipeline":
                return ("result", "run-id", Path("log.jsonl"))
            if name == "ensure_dataframe":
                return pd.DataFrame({"value": [1]})
            return Path("resolved.csv")

        return _inner

    stub._resolve_returns_path = record("resolve_returns_path")  # type: ignore[attr-defined]
    stub._ensure_dataframe = record("ensure_dataframe")  # type: ignore[attr-defined]
    stub._run_pipeline = record("run_pipeline")  # type: ignore[attr-defined]
    stub._print_summary = record("print_summary")  # type: ignore[attr-defined]
    stub._write_report_files = record("write_report_files")  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "trend.cli", stub)

    path, cfg = cli._load_configuration("cfg.yml")
    assert path == Path("cfg.yml")
    assert cfg == {"loaded": True}
    assert cli._resolve_returns_path(Path("cfg.yml"), cfg, None) == Path("resolved.csv")
    df = cli._ensure_dataframe(Path("returns.csv"))
    assert isinstance(df, pd.DataFrame)
    result = cli._run_pipeline(cfg, df, source_path=None, log_file=None, structured_log=True, bundle=None)
    assert result[1] == "run-id"
    cli._print_summary(cfg, result)
    cli._write_report_files(Path("out"), cfg, result, run_id="run")

    names = [name for name, *_ in stub_calls]
    assert names == [
        "resolve_returns_path",
        "ensure_dataframe",
        "run_pipeline",
        "print_summary",
        "write_report_files",
    ]
