"""Tests for :mod:`trend_analysis.proxy.cli`."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from trend_analysis.proxy import cli


def _run_with_argv(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    """Execute ``cli.main`` with ``argv`` patched in ``sys.argv``."""

    monkeypatch.setattr(sys, "argv", argv)
    return cli.main()


def test_proxy_cli_invokes_run_proxy_with_parsed_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = SimpleNamespace(called=False, kwargs={})

    def fake_run_proxy(**kwargs: object) -> None:
        captured.called = True
        captured.kwargs = kwargs

    monkeypatch.setattr(cli, "run_proxy", fake_run_proxy)
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **_: None)

    exit_code = _run_with_argv(
        monkeypatch,
        [
            "proxy-cli",
            "--streamlit-host",
            "streamlit.internal",
            "--streamlit-port",
            "8601",
            "--proxy-host",
            "127.0.0.1",
            "--proxy-port",
            "9000",
            "--log-level",
            "DEBUG",
        ],
    )

    assert exit_code == 0
    assert captured.called is True
    assert captured.kwargs == {
        "streamlit_host": "streamlit.internal",
        "streamlit_port": 8601,
        "proxy_host": "127.0.0.1",
        "proxy_port": 9000,
    }


def test_proxy_cli_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_proxy(**_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_proxy", fake_run_proxy)
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **_: None)

    exit_code = _run_with_argv(monkeypatch, ["proxy-cli"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Proxy server stopped by user" in captured.out


def test_proxy_cli_reports_generic_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_proxy(**_: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_proxy", fake_run_proxy)
    monkeypatch.setattr(cli.logging, "basicConfig", lambda **_: None)

    exit_code = _run_with_argv(monkeypatch, ["proxy-cli"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error starting proxy" in captured.err
    assert "boom" in captured.err
