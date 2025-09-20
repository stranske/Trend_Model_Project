"""Extra runtime coverage for :mod:`trend_analysis.proxy.server`."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from trend_analysis.proxy import server


class _DummyRouter:
    def __init__(self) -> None:
        self.routes: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def add_api_websocket_route(self, *args: Any) -> None:
        self.routes.append(("ws", args, {}))

    def add_api_route(self, *args: Any, **kwargs: Any) -> None:
        self.routes.append(("http", args, kwargs))


class _DummyApp:
    def __init__(self, *_, **__) -> None:
        self.router = _DummyRouter()


class _DummyBackgroundTask:
    def __init__(self, func) -> None:
        self.func = func


class _DummyStreamingResponse:
    def __init__(self, iterator, *, status_code, headers, background):
        self.iterator = iterator
        self.status_code = status_code
        self.headers = headers
        self.background = background


class _DummyAsyncClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    async def request(self, **kwargs: Any):
        self.requests.append(kwargs)

        class _Resp:
            status_code = 204
            headers = {"content-encoding": "gzip", "x-test": "value"}

            async def aiter_bytes(self):
                yield b"payload"

            async def aclose(self):
                return None

        return _Resp()

    async def aclose(self) -> None:
        self.closed = True


def _setup_proxy_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_assert_deps", lambda: None)
    monkeypatch.setattr(server, "FastAPI", _DummyApp)
    monkeypatch.setattr(server, "httpx", SimpleNamespace(AsyncClient=_DummyAsyncClient))
    monkeypatch.setattr(server, "StreamingResponse", _DummyStreamingResponse)
    monkeypatch.setattr(server, "BackgroundTask", _DummyBackgroundTask)


def test_streamlit_proxy_init_requires_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_assert_deps", lambda: None)
    monkeypatch.setattr(server, "FastAPI", None)
    monkeypatch.setattr(server, "httpx", None)

    with pytest.raises(RuntimeError):
        server.StreamlitProxy()


def test_streamlit_proxy_websocket_and_http_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_proxy_deps(monkeypatch)

    dummy_websockets_calls: list[str] = []

    class _DummyTarget:
        async def send(self, _payload):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _DummyConnect:
        def __init__(self, url: str) -> None:
            self.url = url

        async def __aenter__(self):
            dummy_websockets_calls.append(self.url)
            return _DummyTarget()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _DummyWebsockets:
        def connect(self, url: str):
            return _DummyConnect(url)

    monkeypatch.setattr(server, "websockets", _DummyWebsockets())

    gathered: list[tuple[Any, ...]] = []

    async def fake_gather(*coros: Any):
        gathered.append(coros)
        for coro in coros:
            coro.close()
        return None

    monkeypatch.setattr(server.asyncio, "gather", fake_gather)

    proxy = server.StreamlitProxy("example.com", 9000)

    class _DummyClientSocket:
        def __init__(self) -> None:
            self.accepted = False
            self.url = SimpleNamespace(query="token=abc")

        async def accept(self):
            self.accepted = True

        async def close(self, code: int) -> None:  # pragma: no cover - defensive
            self.closed_with = code

        async def receive(self):
            return {"text": "noop"}

        async def send_bytes(self, data: bytes):  # pragma: no cover - defensive
            self.last_bytes = data

        async def send_text(self, data: str):  # pragma: no cover - defensive
            self.last_text = data

    websocket = _DummyClientSocket()

    class _DummyURL:
        def __init__(self, query: str) -> None:
            self.query = query

    class _DummyRequest:
        def __init__(self) -> None:
            self.url = _DummyURL("q=1")
            self.method = "POST"
            self.headers = {"host": "proxy", "x-test": "1"}

        async def body(self) -> bytes:
            return b"payload"

    async def runner() -> None:
        await proxy._websocket_entry(websocket, "foo")
        assert websocket.accepted is True
        expected_url = "ws://example.com:9000/foo?token=abc"
        assert dummy_websockets_calls == [expected_url]
        assert gathered, "asyncio.gather should be invoked"

        response = await proxy._http_entry(_DummyRequest(), "status")
        assert isinstance(response, _DummyStreamingResponse)
        assert dummy_websockets_calls[0] == expected_url

        client = proxy.client
        assert isinstance(client, _DummyAsyncClient)
        assert client.requests
        sent_request = client.requests[0]
        assert sent_request["url"].endswith("/status?q=1")

    asyncio.run(runner())


def test_streamlit_proxy_start_requires_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_proxy_deps(monkeypatch)
    monkeypatch.setattr(server, "uvicorn", None)

    proxy = server.StreamlitProxy()

    async def runner() -> None:
        with pytest.raises(RuntimeError):
            await proxy.start()

    asyncio.run(runner())

