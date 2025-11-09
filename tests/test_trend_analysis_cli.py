from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trend_analysis import cli


class DummyPreset:
    def __init__(self, name: str, payload: dict[str, object]):
        self.name = name
        self._payload = payload

    def as_signal_config(self) -> dict[str, object]:
        return dict(self._payload)


class FrozenConfig:
    def __init__(self) -> None:
        object.__setattr__(self, "signals", {"existing": 1})

    def __setattr__(self, name: str, value) -> None:  # pragma: no cover - defensive
        raise ValueError("frozen config")


def test_load_market_data_csv_defaults(monkeypatch):
    captured = {}

    def fake_load(path: str, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return "loaded"

    monkeypatch.setattr(cli, "load_csv", fake_load)

    result = cli.load_market_data_csv("/tmp/data.csv")

    assert result == "loaded"
    assert captured == {
        "path": "/tmp/data.csv",
        "kwargs": {"errors": "raise", "include_date_column": True},
    }


def test_apply_trend_spec_preset_handles_dict_and_frozen_object():
    preset = DummyPreset("demo", {"window": 10, "lag": 2})
    cfg_dict = {"signals": {"existing": 3}}

    cli._apply_trend_spec_preset(cfg_dict, preset)

    assert cfg_dict["signals"] == {"existing": 3, "window": 10, "lag": 2}
    assert cfg_dict["trend_spec_preset"] == "demo"

    frozen = FrozenConfig()
    cli._apply_trend_spec_preset(frozen, preset)

    assert frozen.signals == {"existing": 1, "window": 10, "lag": 2}
    assert getattr(frozen, "trend_spec_preset") == "demo"


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            {"entries": 1, "hits": 2, "misses": 3, "incremental_updates": 4},
            {"entries": 1, "hits": 2, "misses": 3, "incremental_updates": 4},
        ),
        (
            [
                {"entries": 0, "hits": 0, "misses": 0, "incremental_updates": 0},
                {"entries": 5, "hits": 6, "misses": 7, "incremental_updates": 8},
            ],
            {"entries": 5, "hits": 6, "misses": 7, "incremental_updates": 8},
        ),
        (
            {
                "nested": [
                    {
                        "entries": 1.0,
                        "hits": 2.0,
                        "misses": 3.0,
                        "incremental_updates": 4.0,
                    }
                ]
            },
            {"entries": 1, "hits": 2, "misses": 3, "incremental_updates": 4},
        ),
    ],
)
def test_extract_cache_stats_returns_last_valid(payload, expected):
    assert cli._extract_cache_stats(payload) == expected


def test_check_environment_reports_mismatches(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "requirements.lock"
    lock.write_text("""\n# comment\nalpha==1.0.0\nbeta==2.0.0\ngamma==3.0.0 extra\n""")

    def fake_version(name: str):
        if name == "alpha":
            return "1.0.0"
        if name == "beta":
            raise cli.metadata.PackageNotFoundError(name)
        return "2.0.0"

    monkeypatch.setattr(cli.metadata, "version", fake_version)

    exit_code = cli.check_environment(lock)
    out = capsys.readouterr()

    assert exit_code == 1
    assert "Mismatches detected" in out.out
    assert "beta: installed none" in out.out
    assert "gamma" in out.out


def test_check_environment_success(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "requirements.lock"
    lock.write_text("delta==4.0.0\n")

    monkeypatch.setattr(cli.metadata, "version", lambda name: "4.0.0")

    exit_code = cli.check_environment(lock)
    out = capsys.readouterr()

    assert exit_code == 0
    assert "All packages match" in out.out


def test_maybe_log_step(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        cli, "_log_step", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    cli.maybe_log_step(False, "run", "event", "message")
    cli.maybe_log_step(True, "run", "event", "message", extra=1)

    assert calls == [(("run", "event", "message"), {"extra": 1})]


def test_main_check_flag_invokes_environment(monkeypatch):
    called = {}

    def fake_check(lock_path=None):
        called["checked"] = lock_path
        return 0

    monkeypatch.setattr(cli, "check_environment", fake_check)

    assert cli.main(["--check"]) == 0
    assert "checked" in called


def test_main_gui_invokes_streamlit(monkeypatch):
    class DummyProc:
        returncode = 5

    monkeypatch.setattr(cli.subprocess, "run", lambda args: DummyProc())

    assert cli.main(["gui"]) == 5


def test_main_run_success(monkeypatch, tmp_path, capsys):
    cfg = SimpleNamespace(
        sample_split={
            "in_start": "2020-01-01",
            "in_end": "2020-06-30",
            "out_start": "2020-07-01",
            "out_end": "2020-12-31",
        },
        export={
            "directory": str(tmp_path),
            "formats": ["csv", "excel"],
            "filename": "analysis",
        },
        seed=99,
    )

    spec_preset = DummyPreset("spec", {"kind": "trend"})
    bundle_calls = {}

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "get_trend_spec_preset", lambda name: spec_preset)
    monkeypatch.setattr(cli, "apply_trend_preset", lambda cfg_arg, preset: cfg_arg)
    monkeypatch.setattr(cli, "get_trend_preset", lambda name: "portfolio")
    monkeypatch.setattr(cli, "list_trend_spec_presets", lambda: ["spec"])
    monkeypatch.setattr(cli, "list_preset_slugs", lambda: ["spec"])
    monkeypatch.setattr(
        cli,
        "set_cache_enabled",
        lambda enabled: bundle_calls.setdefault("cache", enabled),
    )

    df = pd.DataFrame({"A": [1, 2, 3]})
    monkeypatch.setattr(cli, "load_market_data_csv", lambda path: df)

    run_result = SimpleNamespace(
        metrics=pd.DataFrame({"m": [1, 2]}),
        details={
            "portfolio_user_weight": pd.Series([0.1, 0.2]),
            "benchmarks": {"bench": pd.Series([1.0, 2.0])},
            "weights_user_weight": pd.Series([0.3, 0.7]),
            "cache": [{"entries": 1, "hits": 2, "misses": 3, "incremental_updates": 4}],
        },
        seed=321,
    )
    monkeypatch.setattr(cli, "run_simulation", lambda cfg_arg, frame: run_result)

    monkeypatch.setattr(
        cli.export, "format_summary_text", lambda *args, **kwargs: "Summary"
    )
    monkeypatch.setattr(
        cli.export, "make_summary_formatter", lambda *args, **kwargs: lambda df: df
    )

    excel_calls: list[tuple] = []
    data_calls: list[tuple] = []
    monkeypatch.setattr(
        cli.export,
        "export_to_excel",
        lambda *args, **kwargs: excel_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        cli.export,
        "export_data",
        lambda *args, **kwargs: data_calls.append((args, kwargs)),
    )

    log_path = tmp_path / "logs.jsonl"
    monkeypatch.setattr(
        cli.run_logging, "get_default_log_path", lambda run_id: log_path
    )
    monkeypatch.setattr(
        cli.run_logging, "init_run_logger", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(cli.run_logging, "log_step", lambda *args, **kwargs: None)

    bundle_target = tmp_path / "bundle.zip"

    def fake_export_bundle(result, target):
        bundle_calls["bundle"] = target

    monkeypatch.setattr(cli, "APP_PATH", Path("/tmp/app.py"))
    monkeypatch.setattr(cli, "LOCK_PATH", tmp_path / "lock.lock")
    monkeypatch.setattr(cli.export.bundle, "export_bundle", fake_export_bundle)

    class DummyProc:
        returncode = 0

    monkeypatch.setattr(cli.subprocess, "run", lambda args: DummyProc())

    monkeypatch.delenv("TREND_SEED", raising=False)

    result = cli.main(
        [
            "run",
            "-c",
            "config.yml",
            "-i",
            "returns.csv",
            "--preset",
            "spec",
            "--bundle",
            str(bundle_target),
        ]
    )

    captured = capsys.readouterr()

    assert result == 0
    assert "Summary" in captured.out
    assert excel_calls and data_calls
    assert bundle_calls["bundle"].name == "bundle.zip"
    assert bundle_calls["cache"] is True


def test_main_run_uses_env_seed(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        sample_split={
            "in_start": "2020-01-01",
            "in_end": "2020-06-30",
            "out_start": "2020-07-01",
            "out_end": "2020-12-31",
        },
        export={},
        seed=5,
    )

    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: None)
    monkeypatch.setattr(
        cli, "load_market_data_csv", lambda path: pd.DataFrame({"A": [1]})
    )
    monkeypatch.setattr(
        cli,
        "run_simulation",
        lambda cfg_arg, frame: SimpleNamespace(
            metrics=pd.DataFrame({"m": [1]}),
            details={},
            seed=cfg_arg.seed,
        ),
    )
    monkeypatch.setattr(
        cli.export, "format_summary_text", lambda *args, **kwargs: "Summary"
    )
    monkeypatch.setattr(cli.export, "export_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli.run_logging, "get_default_log_path", lambda run_id: tmp_path / "log.jsonl"
    )
    monkeypatch.setattr(
        cli.run_logging, "init_run_logger", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(cli.run_logging, "log_step", lambda *args, **kwargs: None)

    monkeypatch.setattr(cli, "APP_PATH", Path("/tmp/app.py"))
    monkeypatch.setattr(
        cli.subprocess, "run", lambda args: SimpleNamespace(returncode=0)
    )

    monkeypatch.setenv("TREND_SEED", "7")

    assert (
        cli.main(
            [
                "run",
                "-c",
                "cfg.yml",
                "-i",
                "returns.csv",
                "--no-cache",
                "--no-structured-log",
            ]
        )
        == 0
    )
    assert cfg.seed == 7


def test_main_run_unknown_preset(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        cli,
        "get_trend_spec_preset",
        lambda name: (_ for _ in ()).throw(KeyError("missing")),
    )
    monkeypatch.setattr(cli, "list_trend_spec_presets", lambda: ["alpha", "beta"])

    exit_code = cli.main(
        ["run", "-c", "cfg.yml", "-i", "returns.csv", "--preset", "missing"]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Unknown preset" in captured.err
    assert "alpha, beta" in captured.err


def test_main_run_market_data_error(monkeypatch, capsys):
    class DummyError(cli.MarketDataValidationError):
        def __init__(self):
            super().__init__("invalid data")

    monkeypatch.setattr(
        cli, "load_config", lambda path: SimpleNamespace(sample_split={}, export={})
    )
    monkeypatch.setattr(cli, "set_cache_enabled", lambda enabled: None)
    monkeypatch.setattr(
        cli, "load_market_data_csv", lambda path: (_ for _ in ()).throw(DummyError())
    )

    exit_code = cli.main(["run", "-c", "cfg.yml", "-i", "returns.csv"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "invalid data" in captured.err
