"""Streamlit proxy (HTTP + WebSocket) with optional dependencies.

This implementation keeps heavy optional libraries (FastAPI, uvicorn,
httpx, websockets) isolated so the rest of the package can be imported
without them. A clear RuntimeError is raised only when starting the
proxy if dependencies are missing. Type checking is satisfied via simple
runtime asserts before use.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Forward declarations for optional dependencies (assigned at runtime)
httpx: Any | None = None
uvicorn: Any | None = None
websockets: Any | None = None
FastAPI: Any | None = None
StreamingResponse: Any | None = None
BackgroundTask: Any | None = None

_DEPS_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - static type hints only
    from httpx import AsyncClient as _HTTPXAsyncClient  # noqa: F401
    from uvicorn import Config as _UvicornConfig  # noqa: F401


def _lazy_import_deps() -> bool:
    """Attempt to import heavy dependencies on-demand.

    Returns True if all imports succeed, False otherwise. This allows
    the module to be imported before the virtual environment is
    activated.
    """
    global httpx, uvicorn, websockets, FastAPI, StreamingResponse, BackgroundTask, _DEPS_AVAILABLE
    try:  # pragma: no cover - side-effect imports
        import importlib

        httpx = importlib.import_module("httpx")
        uvicorn = importlib.import_module("uvicorn")
        websockets = importlib.import_module("websockets")
        fastapi_mod = importlib.import_module("fastapi")
        FastAPI = getattr(fastapi_mod, "FastAPI")
        StreamingResponse = getattr(fastapi_mod.responses, "StreamingResponse")
        starlette_bg = importlib.import_module("starlette.background")
        BackgroundTask = getattr(starlette_bg, "BackgroundTask")
        _DEPS_AVAILABLE = True
        return True
    except Exception:  # pragma: no cover
        _DEPS_AVAILABLE = False
        httpx = None
        uvicorn = None
        websockets = None
        FastAPI = None
        StreamingResponse = None
        BackgroundTask = None
        return False


_DEPS_AVAILABLE = _lazy_import_deps()


def _assert_deps() -> None:
    global _DEPS_AVAILABLE

    # If the modules were explicitly marked as missing (set to None in
    # ``sys.modules``) treat them as unavailable even if they were previously
    # imported.  This mirrors the behaviour expected by the proxy tests where
    # dependencies are monkeypatched out to validate the graceful fallback.
    required = ("fastapi", "uvicorn", "httpx", "websockets")
    explicitly_missing = [
        name for name in required if name in sys.modules and sys.modules[name] is None
    ]
    if explicitly_missing:
        _DEPS_AVAILABLE = False
        raise ImportError(
            "Required dependencies not available (fastapi, uvicorn, httpx, websockets). "
            "Install with: pip install fastapi uvicorn httpx websockets"
        )

    if not _DEPS_AVAILABLE and not _lazy_import_deps():  # pragma: no cover
        raise ImportError(
            "Required dependencies not available (fastapi, uvicorn, httpx, websockets). "
            "Install with: pip install fastapi uvicorn httpx websockets"
        )


@runtime_checkable
class _SupportsWebSocket(Protocol):  # Minimal protocol slice
    async def accept(self) -> None:  # noqa: D401
        ...

    async def close(self, code: int) -> None:  # noqa: D401
        ...

    @property
    def url(self) -> Any:  # noqa: D401
        ...

    async def receive(self) -> dict[str, Any]:  # noqa: D401
        ...

    async def send_bytes(self, data: bytes) -> None:  # noqa: D401
        ...

    async def send_text(self, data: str) -> None:  # noqa: D401
        ...


class StreamlitProxy:
    """Forward HTTP + WebSocket traffic to a Streamlit instance."""

    def __init__(self, streamlit_host: str = "localhost", streamlit_port: int = 8501):
        _assert_deps()
        if FastAPI is None or httpx is None:
            raise RuntimeError("FastAPI/httpx dependencies are required for StreamlitProxy")
        self.streamlit_host = streamlit_host
        self.streamlit_port = streamlit_port
        self.streamlit_base_url = f"http://{streamlit_host}:{streamlit_port}"
        self.streamlit_ws_url = f"ws://{streamlit_host}:{streamlit_port}"
        # FastAPI app + shared async client (lazy optional deps already asserted)
        self.app = FastAPI(title="Streamlit Proxy", version="1.0.0")
        # httpx.AsyncClient typing is fine at runtime after _assert_deps;
        # keep simple Any / inferred type to avoid extra TYPE_CHECKING complexity
        self.client = httpx.AsyncClient()
        self._register_routes()

    def _register_routes(self) -> None:
        self.app.router.add_api_websocket_route("/{path:path}", self._websocket_entry)
        self.app.router.add_api_route(
            "/{path:path}",
            self._http_entry,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        )

    async def _websocket_entry(self, websocket: Any, path: str) -> None:
        await self._handle_websocket(cast(_SupportsWebSocket, websocket), path)

    async def _http_entry(self, request: Any, path: str) -> Any:  # FastAPI Request at runtime
        return await self._handle_http_request(request, path)

    async def _handle_websocket(self, websocket: _SupportsWebSocket, path: str) -> None:
        _assert_deps()
        if websockets is None:
            raise RuntimeError("websockets dependency is required for proxy WebSocket support")
        await websocket.accept()
        target_url = f"{self.streamlit_ws_url}/{path}"
        q = getattr(websocket.url, "query", "")
        if q:
            target_url += f"?{q}"
        logger.info("Proxying WebSocket -> %s", target_url)
        try:  # pragma: no cover
            async with websockets.connect(target_url) as target_ws:

                async def client_to_streamlit() -> None:
                    while True:
                        msg = await websocket.receive()
                        if (b := msg.get("bytes")) is not None:
                            await target_ws.send(b)
                        elif (t := msg.get("text")) is not None:
                            await target_ws.send(t)

                async def streamlit_to_client() -> None:
                    async for payload in target_ws:
                        if isinstance(payload, bytes):
                            await websocket.send_bytes(payload)
                        else:
                            await websocket.send_text(str(payload))

                await asyncio.gather(client_to_streamlit(), streamlit_to_client())
        except Exception as e:  # pragma: no cover
            logger.error("WebSocket proxy error: %s", e)
            await websocket.close(code=1011)

    async def _handle_http_request(self, request: Any, path: str) -> Any:
        _assert_deps()
        normalized = path if path.startswith("/") else f"/{path}"
        target_url = urljoin(self.streamlit_base_url, normalized)
        raw_query = getattr(request.url, "query", "")
        if isinstance(raw_query, bytes):
            query_string = raw_query.decode("utf-8", errors="ignore")
        elif raw_query is None:
            query_string = ""
        else:
            query_string = str(raw_query)
        if query_string:
            target_url += f"?{query_string}"
        logger.debug("Proxying HTTP %s -> %s", getattr(request, "method", "?"), target_url)
        headers = dict(request.headers)
        headers.pop("host", None)
        try:  # pragma: no cover
            body = await request.body()
            response = await self.client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                follow_redirects=True,
            )
            filtered = {
                k: v
                for k, v in response.headers.items()
                if k.lower() not in {"content-encoding", "transfer-encoding"}
            }
            from collections.abc import AsyncIterator

            async def generate() -> AsyncIterator[bytes]:
                async for chunk in response.aiter_bytes():
                    yield chunk

            background = None
            if BackgroundTask is not None:  # pragma: no branch
                background = BackgroundTask(response.aclose)
            if StreamingResponse is None:  # pragma: no cover - defensive
                return {
                    "status_code": response.status_code,
                    "headers": headers,
                    "error": "StreamingResponse dependency missing",
                }
            return StreamingResponse(
                generate(),
                status_code=response.status_code,
                headers=filtered,
                background=background,
            )
        except Exception as e:  # pragma: no cover
            logger.error("HTTP proxy error: %s", e)
            return {"error": str(e), "status_code": 502}

    async def start(self, host: str = "0.0.0.0", port: int = 8500) -> None:  # noqa: D401
        _assert_deps()
        if uvicorn is None:
            raise RuntimeError("uvicorn dependency is required to start the proxy server")
        config = uvicorn.Config(app=self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        logger.info("Starting Streamlit proxy on %s:%s", host, port)
        logger.info("Forwarding to Streamlit at %s", self.streamlit_base_url)
        await server.serve()

    async def close(self) -> None:  # noqa: D401
        _assert_deps()
        await self.client.aclose()


def run_proxy(
    streamlit_host: str = "localhost",
    streamlit_port: int = 8501,
    proxy_host: str = "0.0.0.0",
    proxy_port: int = 8500,
) -> None:
    """Run the proxy synchronously (convenience wrapper)."""

    async def main() -> None:
        proxy = StreamlitProxy(streamlit_host, streamlit_port)
        try:
            await proxy.start(proxy_host, proxy_port)
        finally:
            await proxy.close()

    asyncio.run(main())


__all__ = ["StreamlitProxy", "run_proxy"]

if __name__ == "__main__":  # pragma: no cover
    run_proxy()
