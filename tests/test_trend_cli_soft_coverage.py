"""Focused tests covering lightweight CLI helpers for soft coverage runs."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from trend_analysis import cli
from trend_analysis.signal_presets import get_trend_spec_preset


def test_load_market_data_csv_applies_default_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the legacy shim forwards default arguments to ``load_csv``."""

    captured: dict[str, Any] = {}

    def fake_load_csv(path: str, *, errors: str, include_date_column: bool) -> pd.DataFrame:
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


def test_load_market_data_csv_honours_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def fake_log_step(run_id: str, event: str, message: str, *, level: str, extra: int) -> None:
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
    lock.write_text("numpy==1.26.0\n")

    def fake_version(name: str) -> str:
        if name == "numpy":
            return "1.26.0"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", fake_version)

    exit_code = cli.check_environment(lock)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "All packages match" in out


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

