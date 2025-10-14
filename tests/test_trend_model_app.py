"""Tests for the :mod:`trend_model.app` console entry point."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trend_model import app


@pytest.mark.parametrize(
    "argv, expected_args",
    [
        ([], []),
        (["--server.port", "1234"], ["--server.port", "1234"]),
    ],
)
def test_main_invokes_streamlit(monkeypatch: pytest.MonkeyPatch, argv, expected_args):
    """``main`` should call ``streamlit run`` with the app path and arguments."""

    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return SimpleNamespace(returncode=42)

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    exit_code = app.main(argv)

    assert exit_code == 42
    assert captured["command"] == ["streamlit", "run", str(app.APP_PATH), *expected_args]
    assert captured["check"] is False


def test_main_uses_sys_argv_when_none(monkeypatch: pytest.MonkeyPatch):
    """When ``argv`` is ``None`` the function should forward ``sys.argv[1:]``."""

    fake_args = ["trend-model", "--server.headless=false", "--theme=dark"]
    monkeypatch.setattr(app.sys, "argv", fake_args)

    observed = {}

    def fake_run(command, check):
        observed["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    exit_code = app.main(None)

    assert exit_code == 0
    assert observed["command"] == ["streamlit", "run", str(app.APP_PATH), "--server.headless=false", "--theme=dark"]


def test_main_returns_127_when_streamlit_missing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """A missing ``streamlit`` executable should emit an error and return 127."""

    def raise_file_not_found(command, check):
        raise FileNotFoundError

    monkeypatch.setattr(app.subprocess, "run", raise_file_not_found)

    exit_code = app.main(["--any"])

    captured = capsys.readouterr()

    assert exit_code == 127
    assert "streamlit" in captured.err
    assert "app" in captured.err

