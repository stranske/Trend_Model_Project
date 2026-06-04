from __future__ import annotations

import inspect
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from trend_analysis.llm_proxy import cli as llm_proxy_cli
from trend_analysis.llm_proxy import server as llm_proxy_server
from trend_analysis.proxy import cli as streamlit_proxy_cli
from trend_analysis.proxy import server as streamlit_proxy_server


def test_streamlit_proxy_default_bind_is_localhost(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_proxy(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(streamlit_proxy_cli, "run_proxy", fake_run_proxy)
    monkeypatch.setattr(
        streamlit_proxy_cli, "setup_logging", lambda **_: Path("/tmp/proxy.log")
    )
    monkeypatch.setattr(sys, "argv", ["trend-proxy"])

    assert streamlit_proxy_cli.main() == 0

    assert captured["proxy_host"] == "127.0.0.1"
    assert (
        inspect.signature(streamlit_proxy_server.StreamlitProxy.start)
        .parameters["host"]
        .default
        == "127.0.0.1"
    )
    assert (
        inspect.signature(streamlit_proxy_server.run_proxy)
        .parameters["proxy_host"]
        .default
        == "127.0.0.1"
    )


def test_streamlit_proxy_all_interfaces_requires_explicit_host(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class DummyProxy:
        def __init__(self, streamlit_host: str, streamlit_port: int) -> None:
            events.append(("init", (streamlit_host, streamlit_port)))

        async def start(self, host: str, port: int) -> None:
            events.append(("start", (host, port)))

        async def close(self) -> None:
            events.append(("close", None))

    monkeypatch.setattr(streamlit_proxy_server, "StreamlitProxy", DummyProxy)

    streamlit_proxy_server.run_proxy(proxy_host="0.0.0.0", proxy_port=9000)

    assert ("start", ("0.0.0.0", 9000)) in events


def test_llm_proxy_default_bind_is_localhost(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_proxy(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(llm_proxy_cli, "run_proxy", fake_run_proxy)
    monkeypatch.setattr(llm_proxy_cli, "setup_logging", lambda **_: Path("/tmp/llm.log"))
    monkeypatch.setattr(sys, "argv", ["trend-llm-proxy"])

    assert llm_proxy_cli.main() == 0

    assert captured["host"] == "127.0.0.1"
    assert (
        inspect.signature(llm_proxy_server.LLMProxy.start).parameters["host"].default
        == "127.0.0.1"
    )
    assert (
        inspect.signature(llm_proxy_server.run_proxy).parameters["host"].default
        == "127.0.0.1"
    )


def test_llm_proxy_all_interfaces_requires_explicit_host(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class DummyProxy:
        def __init__(self, upstream_base: str | None = None) -> None:
            events.append(("init", upstream_base))

        async def start(self, *, host: str, port: int) -> None:
            events.append(("start", (host, port)))

        async def close(self) -> None:
            events.append(("close", None))

    monkeypatch.setattr(llm_proxy_server, "LLMProxy", DummyProxy)

    llm_proxy_server.run_proxy(host="0.0.0.0", port=8799)

    assert ("start", ("0.0.0.0", 8799)) in events


def test_api_server_module_entrypoint_default_bind_is_localhost(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_run(*, host: str, port: int) -> None:
        called["host"] = host
        called["port"] = port

    package = ModuleType("trend_analysis.api_server")
    package.run = fake_run  # type: ignore[attr-defined]
    package.__path__ = [str(Path("src/trend_analysis/api_server").resolve())]
    monkeypatch.setitem(sys.modules, "trend_analysis.api_server", package)
    sys.modules.pop("trend_analysis.api_server.__main__", None)

    runpy.run_module("trend_analysis.api_server.__main__", run_name="__main__")

    assert called == {"host": "127.0.0.1", "port": 8000}


def test_api_server_explicit_all_interfaces_override_still_works(monkeypatch) -> None:
    import trend_analysis.api_server as api_server

    calls: list[dict[str, Any]] = []

    class StubUvicorn:
        @staticmethod
        def run(app: Any, host: str, port: int, *, reload: bool, log_level: str) -> None:
            calls.append(
                {
                    "app": app,
                    "host": host,
                    "port": port,
                    "reload": reload,
                    "log_level": log_level,
                }
            )

    monkeypatch.setitem(sys.modules, "uvicorn", StubUvicorn())

    assert api_server.run(host="0.0.0.0", port=1234) == ("0.0.0.0", 1234)
    assert any(call["host"] == "0.0.0.0" and call["port"] == 1234 for call in calls)
