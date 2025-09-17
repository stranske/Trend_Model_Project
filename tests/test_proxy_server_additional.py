"""Additional tests for the Streamlit proxy server helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

import trend_analysis.proxy.server as server


class DummyRouter:
    """Router stub that records registered routes."""

    def __init__(self) -> None:
        self.ws_routes: list[tuple[str, Any]] = []
        self.http_routes: list[tuple[str, Any, tuple[str, ...]]] = []

    def add_api_websocket_route(self, path: str, handler: Any) -> None:
        self.ws_routes.append((path, handler))

    def add_api_route(
        self, path: str, handler: Any, *, methods: list[str]
    ) -> None:
        self.http_routes.append((path, handler, tuple(methods)))


class DummyFastAPI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.router = DummyRouter()


class DummyResponse:
    def __init__(self) -> None:
        self.status_code = 200
        self.headers = {
            "content-encoding": "gzip",
            "transfer-encoding": "chunked",
            "X-Test": "ok",
        }
        self.closed = False

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in (b"chunk-one", b"chunk-two"):
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class DummyAsyncClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        content: bytes,
        follow_redirects: bool,
    ) -> DummyResponse:
        response = DummyResponse()
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "content": content,
                "follow_redirects": follow_redirects,
                "response": response,
            }
        )
        return response

    async def aclose(self) -> None:
        self.closed = True


class DummyBackgroundTask:
    def __init__(self, func: Any) -> None:
        self.func = func


class DummyConfig:
    def __init__(self, *, app: Any, host: str, port: int, log_level: str) -> None:
        self.app = app
        self.host = host
        self.port = port
        self.log_level = log_level


class DummyServer:
    instances: list["DummyServer"] = []

    def __init__(self, config: DummyConfig) -> None:
        self.config = config
        self.served = False
        DummyServer.instances.append(self)

    async def serve(self) -> None:
        self.served = True


def make_streaming_response(
    iterator: AsyncIterator[bytes],
    *,
    status_code: int,
    headers: dict[str, str],
    background: DummyBackgroundTask | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        iterator=iterator,
        status_code=status_code,
        headers=headers,
        background=background,
    )


@pytest.fixture
def patched_server(monkeypatch: pytest.MonkeyPatch) -> Any:
    server._DEPS_AVAILABLE = True
    monkeypatch.setattr(server, "FastAPI", DummyFastAPI)
    monkeypatch.setattr(server, "httpx", SimpleNamespace(AsyncClient=DummyAsyncClient))
    monkeypatch.setattr(server, "StreamingResponse", make_streaming_response)
    monkeypatch.setattr(server, "BackgroundTask", DummyBackgroundTask)
    monkeypatch.setattr(
        server,
        "uvicorn",
        SimpleNamespace(Config=DummyConfig, Server=DummyServer),
    )
    monkeypatch.setattr(server, "websockets", SimpleNamespace(connect=None))
    return server


def test_streamlit_proxy_registers_routes(patched_server: Any) -> None:
    proxy = patched_server.StreamlitProxy("example.com", 1234)
    assert proxy.app.router.ws_routes == [("/{path:path}", proxy._websocket_entry)]
    assert proxy.app.router.http_routes[0][0] == "/{path:path}"
    assert proxy.streamlit_base_url == "http://example.com:1234"
    assert proxy.streamlit_ws_url == "ws://example.com:1234"


def test_handle_http_request_streams_response(patched_server: Any) -> None:
    proxy = patched_server.StreamlitProxy("example.com", 1234)

    class DummyRequest:
        def __init__(self) -> None:
            self.method = "POST"
            self.headers = {"host": "example.com", "x-custom": "value"}
            self.url = SimpleNamespace(query="foo=bar")

        async def body(self) -> bytes:
            return b"payload"

    result = asyncio.run(proxy._handle_http_request(DummyRequest(), "status"))
    assert result.status_code == 200
    # Host header removed but custom headers retained
    recorded = proxy.client.calls[0]
    assert recorded["headers"] == {"x-custom": "value"}
    assert recorded["url"] == "http://example.com:1234/status?foo=bar"
    # Streamed response exposes filtered headers only
    assert result.headers == {"X-Test": "ok"}
    async def collect() -> list[bytes]:
        return [chunk async for chunk in result.iterator]

    chunks = asyncio.run(collect())
    assert chunks == [b"chunk-one", b"chunk-two"]
    assert isinstance(result.background, DummyBackgroundTask)
    response_obj = recorded["response"]
    assert result.background.func == response_obj.aclose


def test_streamlit_proxy_close_closes_httpx_client(patched_server: Any) -> None:
    proxy = patched_server.StreamlitProxy()
    asyncio.run(proxy.close())
    assert proxy.client.closed is True


def test_run_proxy_starts_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[Any] = []

    class DummyProxy:
        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port
            self.started: list[tuple[str, int]] = []
            self.closed = False
            instances.append(self)

        async def start(self, host: str, port: int) -> None:
            self.started.append((host, port))

        async def close(self) -> None:
            self.closed = True

    def fake_run(coro: Any) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(server, "StreamlitProxy", DummyProxy)
    monkeypatch.setattr(server.asyncio, "run", fake_run)

    server.run_proxy("alpha", 1234, proxy_host="0.0.0.0", proxy_port=9000)

    assert len(instances) == 1
    proxy = instances[0]
    assert proxy.host == "alpha"
    assert proxy.port == 1234
    assert proxy.started == [("0.0.0.0", 9000)]
    assert proxy.closed is True
