from __future__ import annotations

import runpy


def test_api_server_entrypoint_invokes_run(monkeypatch):
    called = {}

    def fake_run(*, host: str, port: int) -> None:
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr("trend_analysis.api_server.run", fake_run)

    runpy.run_module("trend_analysis.api_server.__main__", run_name="__main__")

    assert called == {"host": "0.0.0.0", "port": 8000}
