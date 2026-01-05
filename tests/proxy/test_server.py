import asyncio
import logging
import sys
import types
from unittest.mock import AsyncMock

import pytest

from trend_analysis.proxy import server


class DummyRouter:
    def __init__(self) -> None:
        self.websocket_routes = []
        self.http_routes = []

    def add_api_websocket_route(self, path: str, handler: AsyncMock) -> None:
        self.websocket_routes.append((path, handler))

    def add_api_route(self, path: str, handler: AsyncMock, methods: list[str]) -> None:
        self.http_routes.append((path, handler, tuple(methods)))


class DummyApp:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        self.router = DummyRouter()


class DummyBackgroundTask:
    def __init__(self, func) -> None:
        self.func = func


class DummyStreamingResponse:
    def __init__(self, iterable, *, status_code: int, headers: dict[str, str], background) -> None:
        self.iterable = iterable
        self.status_code = status_code
        self.headers = headers
        self.background = background


@pytest.fixture
def proxy_fixture(monkeypatch):
    monkeypatch.setattr(server, "_assert_deps", lambda: None)
    dummy_client = types.SimpleNamespace()
    dummy_client.request = AsyncMock()
    dummy_client.aclose = AsyncMock()
    monkeypatch.setattr(
        server,
        "httpx",
        types.SimpleNamespace(AsyncClient=lambda: dummy_client),
    )
    monkeypatch.setattr(server, "FastAPI", DummyApp)
    monkeypatch.setattr(server, "BackgroundTask", DummyBackgroundTask)
    monkeypatch.setattr(server, "StreamingResponse", DummyStreamingResponse)
    monkeypatch.setattr(server, "websockets", None)
    proxy = server.StreamlitProxy("example.com", 1234)
    return proxy, dummy_client


def test_assert_deps_respects_explicit_missing(monkeypatch):
    monkeypatch.setattr(server, "_DEPS_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "fastapi", None)
    with pytest.raises(ImportError):
        server._assert_deps()


def test_streamlit_proxy_init_requires_dependencies(monkeypatch):
    monkeypatch.setattr(server, "_assert_deps", lambda: None)
    monkeypatch.setattr(server, "FastAPI", None)
    monkeypatch.setattr(server, "httpx", None)
    with pytest.raises(RuntimeError):
        server.StreamlitProxy()


def test_websocket_entry_delegates(proxy_fixture, monkeypatch):
    proxy, _ = proxy_fixture
    handler = AsyncMock()
    monkeypatch.setattr(proxy, "_handle_websocket", handler)
    websocket = object()
    asyncio.run(proxy._websocket_entry(websocket, "path"))
    handler.assert_awaited_once()
    called_args = handler.await_args.args
    assert called_args[0] is websocket
    assert called_args[1] == "path"


def test_http_entry_delegates(proxy_fixture, monkeypatch):
    proxy, _ = proxy_fixture
    handler = AsyncMock()
    monkeypatch.setattr(proxy, "_handle_http_request", handler)
    request = object()
    asyncio.run(proxy._http_entry(request, "endpoint"))
    handler.assert_awaited_once_with(request, "endpoint")


def test_handle_websocket_missing_dependency(proxy_fixture, monkeypatch):
    proxy, _ = proxy_fixture
    monkeypatch.setattr(server, "websockets", None)
    with pytest.raises(RuntimeError):
        asyncio.run(proxy._handle_websocket(types.SimpleNamespace(), "route"))


def test_handle_websocket_query_forwarding(proxy_fixture, monkeypatch, caplog):
    proxy, _ = proxy_fixture

    class DummyWebSocket:
        def __init__(self) -> None:
            self.url = types.SimpleNamespace(query="token=abc")
            self.accept = AsyncMock()
            self.receive = AsyncMock()
            self.send_bytes = AsyncMock()
            self.send_text = AsyncMock()
            self.close = AsyncMock()

    websocket = DummyWebSocket()

    class FailingContext:
        async def __aenter__(self):
            raise RuntimeError("boom")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def connect(url: str):
        return FailingContext()

    monkeypatch.setattr(server, "websockets", types.SimpleNamespace(connect=connect))
    caplog.set_level(logging.INFO)

    asyncio.run(proxy._handle_websocket(websocket, "stream/path"))

    websocket.accept.assert_awaited_once()
    websocket.close.assert_awaited_once()
    assert websocket.close.await_args.kwargs.get("code") == 1011
    assert any(
        "ws://example.com:1234/stream/path?token=abc" in record.getMessage()
        for record in caplog.records
    )


def test_handle_websocket_without_query(proxy_fixture, monkeypatch):
    """When no query string is present the base URL should remain unchanged."""

    proxy, _ = proxy_fixture

    class DummyWebSocket:
        def __init__(self) -> None:
            self.url = types.SimpleNamespace()  # no ``query`` attribute
            self.accept = AsyncMock()
            self.receive = AsyncMock()
            self.send_bytes = AsyncMock()
            self.send_text = AsyncMock()
            self.close = AsyncMock()

    websocket = DummyWebSocket()

    captured: dict[str, str] = {}

    class DummyContext:
        def __init__(self, url: str) -> None:
            self.url = url

        async def __aenter__(self):
            captured["url"] = self.url
            return types.SimpleNamespace()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def connect(url: str):
        return DummyContext(url)

    async def fake_gather(*coroutines):
        for coro in coroutines:
            try:
                coro.close()
            except AttributeError:
                pass
        return None

    monkeypatch.setattr(server, "websockets", types.SimpleNamespace(connect=connect))
    monkeypatch.setattr(server.asyncio, "gather", fake_gather)

    asyncio.run(proxy._handle_websocket(websocket, "plain/path"))

    assert captured["url"] == "ws://example.com:1234/plain/path"
    websocket.accept.assert_awaited_once()
    assert websocket.receive.await_count == 0


class DummyResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "x-extra": "1",
        }
        self.status_code = 204
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def test_handle_http_request_streams_response(proxy_fixture):
    proxy, client = proxy_fixture
    request = types.SimpleNamespace()
    request.method = "POST"
    request.headers = {"host": "ignored", "content-type": "json"}
    request.url = types.SimpleNamespace(query="a=1")
    request.body = AsyncMock(return_value=b"payload")

    response = DummyResponse([b"alpha", b"beta"])
    client.request.return_value = response

    result = asyncio.run(proxy._handle_http_request(request, "api/data"))

    await_args = client.request.await_args
    assert await_args.kwargs["url"] == "http://example.com:1234/api/data?a=1"
    assert await_args.kwargs["headers"] == {"content-type": "json"}
    assert await_args.kwargs["content"] == b"payload"

    assert isinstance(result, DummyStreamingResponse)
    assert result.status_code == response.status_code
    assert result.headers == {"Content-Type": "application/json", "x-extra": "1"}
    assert result.background.func == response.aclose


def test_handle_http_request_without_query(proxy_fixture):
    proxy, client = proxy_fixture
    request = types.SimpleNamespace()
    request.method = "GET"
    request.headers = {"host": "ignored", "accept": "json"}
    request.url = types.SimpleNamespace(query="")
    request.body = AsyncMock(return_value=b"")

    response = DummyResponse([b"chunk"])
    client.request.return_value = response

    result = asyncio.run(proxy._handle_http_request(request, "status"))

    await_args = client.request.await_args
    assert await_args.kwargs["url"] == "http://example.com:1234/status"
    assert isinstance(result, DummyStreamingResponse)


def test_handle_http_request_decodes_bytes_query(proxy_fixture):
    proxy, client = proxy_fixture
    request = types.SimpleNamespace()
    request.method = "GET"
    request.headers = {"host": "ignored"}
    request.url = types.SimpleNamespace(query=b"symbol=%E2%82%AC")
    request.body = AsyncMock(return_value=b"")

    response = DummyResponse([b"chunk"])
    client.request.return_value = response

    asyncio.run(proxy._handle_http_request(request, "feed"))

    target_url = client.request.await_args.kwargs["url"]
    assert target_url.endswith("symbol=%E2%82%AC")


def test_handle_http_request_none_query(proxy_fixture):
    proxy, client = proxy_fixture
    request = types.SimpleNamespace()
    request.method = "GET"
    request.headers = {"host": "ignored"}
    request.url = types.SimpleNamespace(query=None)
    request.body = AsyncMock(return_value=b"")

    response = DummyResponse([b"chunk"])
    client.request.return_value = response

    asyncio.run(proxy._handle_http_request(request, "noop"))

    target_url = client.request.await_args.kwargs["url"]
    assert target_url == "http://example.com:1234/noop"


def test_start_requires_uvicorn(proxy_fixture, monkeypatch):
    proxy, _ = proxy_fixture
    monkeypatch.setattr(server, "uvicorn", None)
    with pytest.raises(RuntimeError):
        asyncio.run(proxy.start("127.0.0.1", 9000))


def test_start_invokes_uvicorn(proxy_fixture, monkeypatch):
    proxy, _ = proxy_fixture

    class DummyConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    created = {}

    class DummyServer:
        def __init__(self, config) -> None:
            self.config = config
            self.serve = AsyncMock(return_value=None)
            created["server"] = self

    monkeypatch.setattr(
        server,
        "uvicorn",
        types.SimpleNamespace(Config=lambda **kwargs: DummyConfig(**kwargs), Server=DummyServer),
    )

    asyncio.run(proxy.start("0.0.0.0", 7777))

    dummy_server = created["server"]
    assert dummy_server.config.kwargs["app"] is proxy.app
    assert dummy_server.config.kwargs["host"] == "0.0.0.0"
    assert dummy_server.config.kwargs["port"] == 7777
    dummy_server.serve.assert_awaited_once()


def test_close_closes_client(proxy_fixture):
    proxy, client = proxy_fixture
    asyncio.run(proxy.close())
    client.aclose.assert_awaited_once()


def test_run_proxy_closes_on_start_failure(monkeypatch):
    events: list[tuple[str, tuple[object, ...]]] = []

    class DummyProxy:
        def __init__(self, *args: object) -> None:
            events.append(("init", args))

        async def start(self, host: str, port: int) -> None:
            events.append(("start", (host, port)))
            raise RuntimeError("boom")

        async def close(self) -> None:
            events.append(("close", ()))

    monkeypatch.setattr(server, "StreamlitProxy", DummyProxy)

    original_run = asyncio.run

    def fake_run(coro):
        return original_run(coro)

    monkeypatch.setattr(server.asyncio, "run", fake_run)

    with pytest.raises(RuntimeError):
        server.run_proxy("alpha", 1234, "beta", 4321)

    assert events[0][0] == "init"
    assert any(name == "close" for name, _ in events)
