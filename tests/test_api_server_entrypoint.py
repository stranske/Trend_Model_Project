from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path


def test_api_server_entrypoint_invokes_run(monkeypatch):
    """Executing the module as ``python -m`` should call ``run`` with the CLI
    defaults."""

    called: dict[str, object] = {}

    def fake_run(*, host: str, port: int) -> None:
        called["host"] = host
        called["port"] = port

    # Patch both the package-level reference and any cached attribute on the
    # ``__main__`` module so the smoke test stays robust even if the module was
    # imported earlier in the test session.
    monkeypatch.setattr("trend_analysis.api_server.run", fake_run)
    monkeypatch.setattr("trend_analysis.api_server.__main__.run", fake_run, raising=False)

    # Ensure ``run_module`` loads a fresh module instance so coverage captures
    # the ``__main__`` guard without runtime warnings about cached modules.
    sys.modules.pop("trend_analysis.api_server.__main__", None)

    runpy.run_module("trend_analysis.api_server.__main__", run_name="__main__")

    assert called == {"host": "0.0.0.0", "port": 8000}


def test_api_server_package_main_invokes_run(monkeypatch):
    """Running the package module directly should trigger the run guard."""

    calls: list[dict[str, object]] = []

    class _StubUvicorn:
        @staticmethod
        def run(app, host: str, port: int, *, reload: bool, log_level: str) -> None:
            calls.append(
                {
                    "app": app,
                    "host": host,
                    "port": port,
                    "reload": reload,
                    "log_level": log_level,
                }
            )

    monkeypatch.setitem(sys.modules, "uvicorn", _StubUvicorn())

    package = importlib.import_module("trend_analysis.api_server")
    module_path = Path(package.__file__)

    runpy.run_path(str(module_path), run_name="__main__")

    assert calls
    assert any(call["host"] == "127.0.0.1" for call in calls)
    assert any(call["port"] == 8000 for call in calls)
    assert any(call["reload"] is False for call in calls)
    assert any(call["log_level"] == "info" for call in calls)
