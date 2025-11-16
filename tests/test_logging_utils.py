from __future__ import annotations

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
