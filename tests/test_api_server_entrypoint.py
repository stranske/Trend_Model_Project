from __future__ import annotations

import runpy
import sys


def test_api_server_entrypoint_invokes_run(monkeypatch):
    """Executing the module as ``python -m`` should call ``run`` with the CLI defaults."""

    called: dict[str, object] = {}

    def fake_run(*, host: str, port: int) -> None:
        called["host"] = host
        called["port"] = port

    # Patch both the package-level reference and any cached attribute on the
    # ``__main__`` module so the smoke test stays robust even if the module was
    # imported earlier in the test session.
    monkeypatch.setattr("trend_analysis.api_server.run", fake_run)
    monkeypatch.setattr(
        "trend_analysis.api_server.__main__.run", fake_run, raising=False
    )

    # Ensure ``run_module`` loads a fresh module instance so coverage captures
    # the ``__main__`` guard without runtime warnings about cached modules.
    sys.modules.pop("trend_analysis.api_server.__main__", None)

    runpy.run_module("trend_analysis.api_server.__main__", run_name="__main__")

    assert called == {"host": "0.0.0.0", "port": 8000}
