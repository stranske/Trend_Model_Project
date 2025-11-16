from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


def test_proxy_cli_main_invocation(monkeypatch, capsys):
    args = ["trend-proxy", "--streamlit-port", "1234", "--proxy-port", "5678"]
    monkeypatch.setattr(sys, "argv", args)

    call_args: dict[str, object] = {}

    def fake_run_proxy(**kwargs):
        call_args.update(kwargs)

    stub_server = ModuleType("trend_analysis.proxy.server")
    stub_server.run_proxy = fake_run_proxy
    stub_server.StreamlitProxy = object
    monkeypatch.setitem(sys.modules, "trend_analysis.proxy.server", stub_server)
    monkeypatch.setattr(
        "trend_analysis.proxy.cli.setup_logging", lambda **_: Path("/tmp/proxy.log")
    )

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("trend_analysis.proxy.cli", run_name="__main__")

    assert exc.value.code == 0
    assert call_args == {
        "streamlit_host": "localhost",
        "streamlit_port": 1234,
        "proxy_host": "0.0.0.0",
        "proxy_port": 5678,
    }
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Proxy CLI logs stored at" in captured.err
