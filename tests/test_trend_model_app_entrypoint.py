"""Tests for the lightweight ``trend-app`` console entry point."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import trend_model.app as trend_app


def test_main_invokes_streamlit_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper should shell out to ``streamlit run`` with our app path."""

    invoked: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> types.SimpleNamespace:
        invoked["cmd"] = cmd
        return types.SimpleNamespace(returncode=11)

    monkeypatch.setattr(trend_app.subprocess, "run", fake_run)

    code = trend_app.main(["--", "--server.port", "9999"])

    assert code == 11
    command = invoked["cmd"]
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert Path(command[4]).name == "app.py"
    # Extra flags after ``--`` should be forwarded untouched.
    assert command[5:] == ["--server.port", "9999"]

