from __future__ import annotations

import sys
from pathlib import Path

from trend_analysis import script_logging as logging_utils


def test_setup_script_logging_calls_underlying_helper(monkeypatch, capsys):
    captured = {}

    def fake_setup_logging(*, app_name: str) -> Path:
        captured["app_name"] = app_name
        return Path(f"/tmp/{app_name}.log")

    monkeypatch.setenv("TREND_DISABLE_PERF_LOGS", "")
    monkeypatch.setattr(
        logging_utils,
        "setup_logging",
        fake_setup_logging,
    )

    log_path = logging_utils.setup_script_logging(module_file="/tmp/demo_alpha.py")

    assert log_path == Path("/tmp/demo-alpha.log")
    assert captured["app_name"] == "demo-alpha"
    out = capsys.readouterr().out
    assert "demo-alpha" in out
    assert str(log_path) in out


def test_setup_script_logging_honours_explicit_app_and_silences_announce(
    monkeypatch, capsys
):
    captured = {}

    def fake_setup_logging(*, app_name: str) -> Path:
        captured["app_name"] = app_name
        return Path(f"/tmp/{app_name}.log")

    monkeypatch.setenv("TREND_DISABLE_PERF_LOGS", "")
    monkeypatch.setattr(logging_utils, "setup_logging", fake_setup_logging)

    log_path = logging_utils.setup_script_logging(
        app_name="custom-app",
        module_file=None,
        announce=False,
    )

    assert log_path == Path("/tmp/custom-app.log")
    assert captured["app_name"] == "custom-app"
    assert capsys.readouterr().out == ""


def test_setup_script_logging_respects_disable(monkeypatch):
    monkeypatch.setenv("TREND_DISABLE_PERF_LOGS", "true")
    calls = []

    def fake_setup_logging(*, app_name: str) -> Path:  # pragma: no cover - guard
        calls.append(app_name)
        return Path("/tmp/ignored.log")

    monkeypatch.setattr(logging_utils, "setup_logging", fake_setup_logging)
    assert logging_utils.setup_script_logging(module_file="/tmp/foo.py") is None
    assert calls == []


def test_run_with_script_logging_executes_callable(monkeypatch):
    order: list[str] = []

    def fake_setup_logging(*, app_name: str) -> Path:
        order.append(app_name)
        return Path("/tmp/sample.log")

    monkeypatch.setenv("TREND_DISABLE_PERF_LOGS", "")
    monkeypatch.setattr(logging_utils, "setup_logging", fake_setup_logging)

    def sample(arg: int) -> int:
        order.append(f"run:{arg}")
        return arg + 1

    result = logging_utils.run_with_script_logging(sample, 2, module_file="/tmp/foo.py")

    assert result == 3
    assert order[0] == "foo"
    assert order[1] == "run:2"


def test_derive_app_name_falls_back_to_sys_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["/tmp/fallback_script.py"])
    monkeypatch.setenv("TREND_DISABLE_PERF_LOGS", "")

    calls: list[str] = []

    def fake_setup_logging(*, app_name: str) -> Path:
        calls.append(app_name)
        return Path("/tmp/fallback.log")

    monkeypatch.setattr(logging_utils, "setup_logging", fake_setup_logging)

    logging_utils.setup_script_logging(module_file=None)

    assert calls == ["fallback-script"]
