"""Focused tests covering lightweight CLI helpers for soft coverage runs."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis import cli
from trend_analysis.signal_presets import get_trend_spec_preset


class StubSpecPreset:
    def __init__(self, name: str = "Balanced", window: int = 14) -> None:
        self.name = name
        self.spec = SimpleNamespace(window=window)

    def as_signal_config(self) -> dict[str, int]:
        return {"window": self.spec.window}


class RejectRunIdNamespace(SimpleNamespace):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "run_id" and name not in self.__dict__:
            raise RuntimeError("run_id immutable")
        super().__setattr__(name, value)


def test_load_market_data_csv_applies_default_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the legacy shim forwards default arguments to ``load_csv``."""

    captured: dict[str, Any] = {}

    def fake_load_csv(
        path: str, *, errors: str, include_date_column: bool
    ) -> pd.DataFrame:
        captured["path"] = path
        captured["errors"] = errors
        captured["include_date_column"] = include_date_column
        return pd.DataFrame({"Date": pd.date_range("2024-01-31", periods=2, freq="ME")})

    monkeypatch.setattr(cli, "load_csv", fake_load_csv)

    frame = cli.load_market_data_csv("sample.csv")

    assert captured == {
        "path": "sample.csv",
        "errors": "raise",
        "include_date_column": True,
    }
    assert isinstance(frame, pd.DataFrame)


def test_load_market_data_csv_honours_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit keyword overrides should be forwarded unchanged."""

    received: dict[str, Any] = {}

    def fake_load_csv(
        path: str,
        *,
        errors: str,
        include_date_column: bool,
        custom_flag: str,
    ) -> pd.DataFrame:
        received.update(
            {
                "path": path,
                "errors": errors,
                "include_date_column": include_date_column,
                "custom_flag": custom_flag,
            }
        )
        return pd.DataFrame({"Date": pd.date_range("2024-02-29", periods=1, freq="ME")})

    monkeypatch.setattr(cli, "load_csv", fake_load_csv)

    cli.load_market_data_csv(
        "custom.csv",
        errors="ignore",
        include_date_column=False,
        custom_flag="sentinel",
    )

    assert received == {
        "path": "custom.csv",
        "errors": "ignore",
        "include_date_column": False,
        "custom_flag": "sentinel",
    }


def test_apply_trend_spec_preset_updates_mapping() -> None:
    """Merging a preset into a plain mapping should update signals in place."""

    preset = get_trend_spec_preset("Balanced")
    cfg: dict[str, Any] = {"signals": {"alpha": 1}}

    cli._apply_trend_spec_preset(cfg, preset)

    assert cfg["trend_spec_preset"] == preset.name
    for key, value in preset.as_signal_config().items():
        assert cfg["signals"][key] == value


class ValueGuardNamespace:
    """Object that raises ValueError when guarded attributes are assigned."""

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "signals", dict(initial or {}))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"signals", "trend_spec_preset"}:
            raise ValueError("guarded attribute")
        object.__setattr__(self, name, value)


def test_apply_trend_spec_preset_updates_guarded_object() -> None:
    preset = get_trend_spec_preset("Conservative")
    cfg = ValueGuardNamespace()

    cli._apply_trend_spec_preset(cfg, preset)

    assert cfg.signals["window"] == preset.spec.window
    assert cfg.trend_spec_preset == preset.name


def test_log_step_delegates_to_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_log_step(
        run_id: str, event: str, message: str, *, level: str, extra: int
    ) -> None:
        captured.update(
            {
                "run_id": run_id,
                "event": event,
                "message": message,
                "level": level,
                "extra": extra,
            }
        )

    monkeypatch.setattr(cli.run_logging, "log_step", fake_log_step)

    cli._log_step("abc123", "stage", "hello", level="DEBUG", extra=7)

    assert captured == {
        "run_id": "abc123",
        "event": "stage",
        "message": "hello",
        "level": "DEBUG",
        "extra": 7,
    }


def test_extract_cache_stats_returns_latest_integerised_snapshot() -> None:
    payload = {
        "first": {
            "entries": 1,
            "hits": 0,
            "misses": 0,
            "incremental_updates": 0,
        },
        "history": [
            {
                "entries": 5,
                "hits": 2.0,  # float values should be coerced to ints when integral
                "misses": 1,
                "incremental_updates": 3.0,
            },
            {
                "entries": 7,
                "hits": 4,
                "misses": 2,
                "incremental_updates": 6,
            },
        ],
    }

    stats = cli._extract_cache_stats(payload)

    assert stats == {"entries": 7, "hits": 4, "misses": 2, "incremental_updates": 6}


def test_extract_cache_stats_skips_non_integral_records() -> None:
    payload = {
        "history": [
            {
                "entries": 1,
                "hits": 2,
                "misses": "oops",
                "incremental_updates": 0,
            },
            {
                "entries": 3.0,
                "hits": 1.0,
                "misses": 0.0,
                "incremental_updates": 2.0,
            },
        ]
    }

    stats = cli._extract_cache_stats(payload)

    assert stats == {"entries": 3, "hits": 1, "misses": 0, "incremental_updates": 2}


def test_check_environment_reports_mismatches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("pandas==1.5.0\nunknown==0.1\n")

    def fake_version(name: str) -> str:
        if name == "pandas":
            return "1.4.9"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", fake_version)

    exit_code = cli.check_environment(lock)
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Mismatches detected" in out
    assert "pandas" in out
    assert "unknown" in out


def test_check_environment_success_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text("# comment\n\ninvalid\nnumpy==1.26.0\n")

    def fake_version(name: str) -> str:
        if name == "numpy":
            return "1.26.0"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", fake_version)

    exit_code = cli.check_environment(lock)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "All packages match" in out


def test_check_environment_missing_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "absent.lock"

    exit_code = cli.check_environment(missing)
    out = capsys.readouterr().out

    assert exit_code == 1
    assert f"Lock file not found: {missing}" in out


def test_maybe_log_step_invokes_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    def fake_log_step(run_id: str, event: str, message: str) -> None:
        called["payload"] = (run_id, event, message)

    monkeypatch.setattr(cli, "_log_step", fake_log_step)

    cli.maybe_log_step(True, "run", "stage", "message")
    assert called["payload"] == ("run", "stage", "message")

    called.clear()
    cli.maybe_log_step(False, "run", "stage", "message")
    assert called == {}


def test_apply_trend_spec_preset_handles_namespace() -> None:
    preset = get_trend_spec_preset("Aggressive")
    cfg = SimpleNamespace(signals={"beta": 2})

    cli._apply_trend_spec_preset(cfg, preset)

    assert cfg.signals["window"] == preset.spec.window
    assert cfg.trend_spec_preset == preset.name


def test_main_handles_check_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_codes: list[int] = []

    def fake_check() -> int:
        exit_codes.append(99)
        return 99

    monkeypatch.setattr(cli, "check_environment", fake_check)

    result = cli.main(["--check"])

    assert result == 99
    assert exit_codes == [99]


def test_main_invokes_streamlit_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class DummyResult:
        returncode = 17

    def fake_run(args: list[str]) -> DummyResult:
        calls.append(args)
        return DummyResult()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.main(["gui"])

    assert result == 17
    assert calls == [["streamlit", "run", str(cli.APP_PATH)]]


def test_main_check_option_after_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_check() -> int:
        calls.append(1)
        return 7

    monkeypatch.setattr(cli, "check_environment", fake_check)

    result = cli.main(
        [
            "run",
            "--config",
            "config.yml",
            "--input",
            "returns.csv",
            "--check",
        ]
    )

    assert result == 7
    assert calls == [1]


def test_main_check_branch_via_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_parse_args(
        self: Any, args: Any | None = None, namespace: Any | None = None
    ) -> Any:
        return SimpleNamespace(check=True, command="noop")

    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(cli, "check_environment", lambda: calls.append(1) or 5)

    result = cli.main([])

    assert result == 5
    assert calls == [1]


def test_main_returns_zero_for_unknown_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_parse_args(
        self: Any, args: Any | None = None, namespace: Any | None = None
    ) -> Any:
        return SimpleNamespace(check=False, command="noop")

    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", fake_parse_args)

    assert cli.main([]) == 0


def test_main_run_rejects_unknown_trend_preset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = SimpleNamespace()

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)

    def fake_get_trend_spec_preset(name: str) -> SimpleNamespace:
        raise KeyError(name)

    monkeypatch.setattr(cli, "get_trend_spec_preset", fake_get_trend_spec_preset)
    monkeypatch.setattr(
        cli, "list_trend_spec_presets", lambda: ["Conservative", "Balanced"]
    )

    exit_code = cli.main(
        [
            "run",
            "--config",
            "config.yml",
            "--input",
            "returns.csv",
            "--preset",
            "unknown",
        ]
    )

    captured = capsys.readouterr().err

    assert exit_code == 2
    assert "Unknown preset" in captured
    assert "Conservative" in captured


def test_main_run_handles_market_data_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = SimpleNamespace(sample_split={}, export={})

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: None)

    def fake_load_csv(path: str) -> None:
        raise cli.MarketDataValidationError("invalid returns file")

    monkeypatch.setattr(cli, "load_market_data_csv", fake_load_csv)

    exit_code = cli.main(
        [
            "run",
            "--config",
            "config.yml",
            "--input",
            "returns.csv",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "invalid returns file" in captured.err


def test_main_run_uses_env_seed_and_legacy_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TREND_SEED", "123")

    cfg = RejectRunIdNamespace(
        sample_split={"in_start": "2020-01-01"}, export={}, seed=5
    )

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)

    cache_flags: list[bool] = []
    monkeypatch.setattr(cli, "set_cache_enabled", lambda flag: cache_flags.append(flag))

    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: pd.DataFrame({"A": [0.1, 0.2]}),
    )
    monkeypatch.setattr(
        cli.pipeline, "run", lambda cfg_obj: pd.DataFrame({"Sharpe": [0.5]})
    )
    monkeypatch.setattr(cli.pipeline, "run_full", lambda cfg_obj: {})
    monkeypatch.setattr(
        cli.run_logging,
        "get_default_log_path",
        lambda run_id: tmp_path / f"{run_id}.jsonl",
    )
    init_calls: list[Any] = []
    monkeypatch.setattr(
        cli.run_logging, "init_run_logger", lambda run_id, path: init_calls.append(path)
    )
    log_calls: list[Any] = []
    monkeypatch.setattr(
        cli.run_logging, "log_step", lambda *args, **kwargs: log_calls.append(args)
    )

    import uuid as uuid_module

    monkeypatch.setattr(
        uuid_module, "uuid4", lambda: SimpleNamespace(hex="1234567890abcdef")
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trend-model",
            "run",
            "--config",
            "config.yml",
            "--input",
            "returns.csv",
            "--no-structured-log",
        ],
    )

    exit_code = cli.main()
    output = capsys.readouterr()

    assert exit_code == 0
    assert "No results" in output.out
    assert cache_flags == [True]
    assert cfg.seed == 123
    assert init_calls == []
    assert log_calls == []


def test_main_run_rejects_unknown_portfolio_preset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = SimpleNamespace(sample_split={}, export={})

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda flag: None)

    monkeypatch.setattr(cli, "get_trend_spec_preset", lambda name: StubSpecPreset())

    def fake_get_trend_preset(name: str) -> Any:
        raise KeyError(name)

    monkeypatch.setattr(cli, "get_trend_preset", fake_get_trend_preset)
    monkeypatch.setattr(cli, "list_preset_slugs", lambda: ["Alpha", "Beta"])

    exit_code = cli.main(
        [
            "run",
            "--config",
            "config.yml",
            "--input",
            "returns.csv",
            "--preset",
            "Alpha",
        ]
    )

    captured = capsys.readouterr().err

    assert exit_code == 2
    assert "Available" in captured
    assert "Alpha" in captured and "Beta" in captured


def test_main_run_default_outputs_and_bundle_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = SimpleNamespace(
        sample_split={
            "in_start": "2022-01-01",
            "in_end": "2022-12-31",
            "out_start": "2023-01-01",
            "out_end": "2023-12-31",
        },
        export={},
        seed=5,
    )

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)

    cache_flags: list[bool] = []
    monkeypatch.setattr(cli, "set_cache_enabled", lambda flag: cache_flags.append(flag))

    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: pd.DataFrame({"A": [0.1, 0.2]}),
    )

    details = {
        "history": [
            {"entries": 2, "hits": 1, "misses": 0, "incremental_updates": 1},
            {"entries": 4, "hits": 3, "misses": 0, "incremental_updates": 2},
        ],
        "portfolio_equal_weight": [0.5, 0.5],
        "benchmarks": {"core": [0.1, 0.2]},
        "weights_user_weight": pd.DataFrame({"A": [1.0]}),
    }
    run_result = SimpleNamespace(
        metrics=pd.DataFrame({"Sharpe": [0.9]}), details=details, seed=21
    )

    monkeypatch.setattr(cli, "get_trend_spec_preset", lambda name: StubSpecPreset())
    monkeypatch.setattr(
        cli, "get_trend_preset", lambda name: SimpleNamespace(name=name)
    )
    monkeypatch.setattr(cli, "apply_trend_preset", lambda cfg_obj, preset: None)

    monkeypatch.setattr(cli, "run_simulation", lambda cfg_obj, frame: run_result)

    monkeypatch.setattr(
        cli.export,
        "format_summary_text",
        lambda *args: "Summary text",
    )
    monkeypatch.setattr(
        cli.export, "make_summary_formatter", lambda *args: lambda *_: None
    )

    exports: list[tuple[str, Any]] = []

    def fake_export_to_excel(data: dict[str, Any], path: str, **_: Any) -> None:
        exports.append(("excel", path))

    def fake_export_data(data: dict[str, Any], path: str, formats: list[str]) -> None:
        exports.append(("data", (path, tuple(formats))))

    monkeypatch.setattr(cli.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(cli.export, "export_data", fake_export_data)

    bundles: list[Path] = []
    from trend_analysis.export import bundle as bundle_mod

    monkeypatch.setattr(
        bundle_mod, "export_bundle", lambda result, path: bundles.append(path)
    )

    monkeypatch.setattr(
        cli.run_logging,
        "get_default_log_path",
        lambda run_id: tmp_path / f"log_{run_id}.jsonl",
    )
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda run_id, path: None)

    log_events: list[str] = []

    def fake_log_step(run_id: str, event: str, message: str, **fields: Any) -> None:
        log_events.append(event)

    monkeypatch.setattr(cli.run_logging, "log_step", fake_log_step)

    import uuid as uuid_module

    monkeypatch.setattr(
        uuid_module, "uuid4", lambda: SimpleNamespace(hex="abcdef1234567890")
    )

    bundle_dir = tmp_path / "bundle_dir"
    bundle_dir.mkdir()

    exit_code = cli.main(
        [
            "run",
            "--config",
            "config.yml",
            "--input",
            "returns.csv",
            "--preset",
            "Balanced",
            "--no-cache",
            "--bundle",
            str(bundle_dir),
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Summary text" in output
    assert cache_flags == [False]
    assert exports[0][0] == "excel"
    assert exports[1][0] == "data"
    assert bundles == [bundle_dir / "analysis_bundle.zip"]
    assert "export_complete" in log_events
    assert cfg.run_id == "abcdef123456"
    assert getattr(run_result, "portfolio") == [0.5, 0.5]
    assert getattr(run_result, "benchmark") == [0.1, 0.2]


def test_main_run_legacy_bundle_builds_run_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = RejectRunIdNamespace(
        sample_split={"in_start": "2020-01-01"},
        export={"directory": str(tmp_path), "formats": ["json"]},
        seed=2,
    )

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda flag: None)

    monkeypatch.setattr(
        cli,
        "load_market_data_csv",
        lambda path: pd.DataFrame({"A": [0.1, 0.2]}),
    )
    monkeypatch.setattr(
        cli.pipeline, "run", lambda cfg_obj: pd.DataFrame({"Sharpe": [0.2]})
    )
    monkeypatch.setattr(
        cli.pipeline,
        "run_full",
        lambda cfg_obj: {
            "history": [
                {"entries": 1, "hits": 1, "misses": 0, "incremental_updates": 0},
                {"entries": 2, "hits": 2, "misses": 0, "incremental_updates": 1},
            ]
        },
    )

    monkeypatch.setattr(
        cli.export,
        "format_summary_text",
        lambda *args: "Legacy summary",
    )
    monkeypatch.setattr(
        cli.export, "make_summary_formatter", lambda *args: lambda *_: None
    )
    monkeypatch.setattr(cli.export, "export_to_excel", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.export, "export_data", lambda *args, **kwargs: None)

    from trend_analysis.export import bundle as bundle_mod

    bundles: list[Path] = []

    monkeypatch.setattr(
        bundle_mod, "export_bundle", lambda result, path: bundles.append(path)
    )

    monkeypatch.setattr(
        cli.run_logging,
        "get_default_log_path",
        lambda run_id: tmp_path / f"log_{run_id}.jsonl",
    )
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda run_id, path: None)
    monkeypatch.setattr(cli.run_logging, "log_step", lambda *args, **kwargs: None)

    import uuid as uuid_module

    monkeypatch.setattr(
        uuid_module, "uuid4", lambda: SimpleNamespace(hex="feedfacecafebeef")
    )

    exit_code = cli.main(
        [
            "run",
            "--config",
            "config.yml",
            "--input",
            "returns.csv",
            "--bundle",
        ]
    )

    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Legacy summary" in output
    assert "Bundle written" in output
    assert bundles == [Path("analysis_bundle.zip")]
    assert getattr(cfg, "run_id", None) is None


def test_unified_loader_wrappers_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    def fake_load(path: str) -> tuple[Path, Any]:
        called["load"] = path
        return Path("cfg.yml"), {"loaded": True}

    def fake_resolve(config_path: Path, cfg: Any, override: str | None) -> Path:
        called["resolve"] = (config_path, override)
        return config_path / "returns.csv"

    def fake_ensure(path: Path) -> pd.DataFrame:
        called["ensure"] = path
        return pd.DataFrame()

    def fake_run_pipeline(*args: Any, **kwargs: Any) -> tuple[int, str, Path | None]:
        called["run_pipeline"] = (args, kwargs)
        return 1, "ok", None

    def fake_print_summary(cfg: Any, result: Any) -> None:
        called["print_summary"] = (cfg, result)

    def fake_write_reports(
        out_dir: Path, cfg: Any, result: Any, *, run_id: str
    ) -> None:
        called["write_reports"] = (out_dir, run_id)

    monkeypatch.setattr("trend.cli._load_configuration", fake_load)
    monkeypatch.setattr("trend.cli._resolve_returns_path", fake_resolve)
    monkeypatch.setattr("trend.cli._ensure_dataframe", fake_ensure)
    monkeypatch.setattr("trend.cli._run_pipeline", fake_run_pipeline)
    monkeypatch.setattr("trend.cli._print_summary", fake_print_summary)
    monkeypatch.setattr("trend.cli._write_report_files", fake_write_reports)

    cfg_path, payload = cli._load_configuration("settings.yml")
    resolved = cli._resolve_returns_path(Path("config.yml"), payload, "returns.csv")
    ensured = cli._ensure_dataframe(Path("returns.csv"))
    pipeline_result = cli._run_pipeline(
        "cfg",
        ensured,
        source_path=None,
        log_file=None,
        structured_log=False,
        bundle=None,
    )
    cli._print_summary("cfg", pipeline_result)
    cli._write_report_files(Path("out"), "cfg", pipeline_result, run_id="xyz")

    assert called["load"] == "settings.yml"
    assert cfg_path == Path("cfg.yml")
    assert resolved == Path("config.yml") / "returns.csv"
    assert "ensure" in called and isinstance(ensured, pd.DataFrame)
    assert pipeline_result == (1, "ok", None)
    assert called["print_summary"][1] == pipeline_result
    assert called["write_reports"] == (Path("out"), "xyz")


def test_main_run_executes_pipeline_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TREND_SEED", raising=False)

    class FakeTrendSpecPreset:
        name = "Balanced"
        spec = SimpleNamespace(window=14)

        def as_signal_config(self) -> dict[str, int]:
            return {"window": 14}

    cfg = SimpleNamespace(
        sample_split={
            "in_start": "2020-01-01",
            "in_end": "2020-12-31",
            "out_start": "2021-01-01",
            "out_end": "2021-12-31",
        },
        export={
            "directory": str(tmp_path),
            "formats": ["Excel"],
            "filename": "analysis",
        },
        seed=5,
    )

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        cli, "get_trend_spec_preset", lambda name: FakeTrendSpecPreset()
    )
    monkeypatch.setattr(
        cli, "get_trend_preset", lambda name: SimpleNamespace(name=name)
    )

    applied_presets: list[tuple[Any, Any]] = []
    monkeypatch.setattr(
        cli,
        "apply_trend_preset",
        lambda target, preset: applied_presets.append((target, preset)),
    )

    cache_flags: list[bool] = []
    monkeypatch.setattr(cli, "set_cache_enabled", lambda flag: cache_flags.append(flag))

    def fake_load_market_data_csv(path: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.date_range("2022-01-31", periods=3, freq="ME"),
                "FundA": [0.1, 0.2, 0.3],
            }
        )

    monkeypatch.setattr(cli, "load_market_data_csv", fake_load_market_data_csv)

    metrics_df = pd.DataFrame({"Sharpe": [1.23]})
    details = {
        "history": [
            {"entries": 1, "hits": 0, "misses": 0, "incremental_updates": 0},
            {"entries": 5, "hits": 2, "misses": 1, "incremental_updates": 3},
        ],
        "portfolio_user_weight": None,
        "portfolio_equal_weight": [0.6, 0.4],
        "benchmarks": {"core": [0.1, 0.2]},
        "weights_user_weight": pd.DataFrame({"FundA": [1.0]}),
    }
    run_result = SimpleNamespace(metrics=metrics_df, details=details, seed=11)

    simulation_calls: list[Any] = []

    def fake_run_simulation(cfg_obj: Any, frame: pd.DataFrame) -> Any:
        simulation_calls.append((cfg_obj, frame))
        return run_result

    monkeypatch.setattr(cli, "run_simulation", fake_run_simulation)

    summary_calls: list[Any] = []
    monkeypatch.setattr(
        cli.export,
        "format_summary_text",
        lambda res, a, b, c, d: summary_calls.append((res, a, b, c, d))
        or "Summary text",
    )

    monkeypatch.setattr(
        cli.export,
        "make_summary_formatter",
        lambda *args: lambda *_: None,
    )

    exported: list[tuple[str, Any, Any]] = []

    def fake_export_to_excel(data: dict[str, Any], path: str, **kwargs: Any) -> None:
        exported.append(("excel", path, sorted(data)))

    def fake_export_data(data: dict[str, Any], path: str, formats: list[str]) -> None:
        exported.append(("data", path, formats))

    monkeypatch.setattr(cli.export, "export_to_excel", fake_export_to_excel)
    monkeypatch.setattr(cli.export, "export_data", fake_export_data)

    bundles: list[tuple[Any, Path]] = []
    from trend_analysis.export import bundle as bundle_mod

    def fake_export_bundle(result: Any, bundle_path: Path) -> None:
        bundles.append((result, bundle_path))

    monkeypatch.setattr(bundle_mod, "export_bundle", fake_export_bundle)

    log_steps: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cli.run_logging,
        "get_default_log_path",
        lambda run_id: tmp_path / f"log_{run_id}.jsonl",
    )
    monkeypatch.setattr(cli.run_logging, "init_run_logger", lambda run_id, path: None)

    def fake_log_step(run_id: str, event: str, message: str, **fields: Any) -> None:
        log_steps.append((event, message))

    monkeypatch.setattr(cli.run_logging, "log_step", fake_log_step)

    import uuid as uuid_module

    monkeypatch.setattr(
        uuid_module, "uuid4", lambda: SimpleNamespace(hex="abcdef1234567890")
    )

    bundle_path = tmp_path / "bundle.zip"
    log_file = tmp_path / "explicit-log.jsonl"

    exit_code = cli.main(
        [
            "run",
            "--config",
            "config.yml",
            "--input",
            "returns.csv",
            "--preset",
            "Balanced",
            "--seed",
            "77",
            "--bundle",
            str(bundle_path),
            "--log-file",
            str(log_file),
        ]
    )

    output = capsys.readouterr()

    assert exit_code == 0
    assert cache_flags == [True]
    assert simulation_calls and simulation_calls[0][0] is cfg
    assert summary_calls and summary_calls[0][0] is details
    assert "Summary text" in output.out
    assert "Cache statistics" in output.out
    assert f"Bundle written: {bundle_path}" in output.out
    assert bundles and bundles[0] == (run_result, bundle_path)
    assert any(event == "export_complete" for event, _ in log_steps)
    assert cfg.seed == 77
    assert applied_presets and applied_presets[0][0] is cfg
    assert exported[0][0] == "excel"
    assert exported[1] == (
        "data",
        str(Path(cfg.export["directory"]) / cfg.export["filename"]),
        ["Excel"],
    )
