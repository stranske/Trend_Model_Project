"""OpenAI-compatible LLM proxy for shared deployments."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import secrets
from typing import Any
from urllib.parse import urljoin

FastAPI: Any | None = None
HTTPException: Any | None = None
Request: Any | None = None
Response: Any | None = None
httpx: Any | None = None
uvicorn: Any | None = None

logger = logging.getLogger(__name__)
_DEPS_AVAILABLE = False


def _load_deps() -> None:
    global FastAPI, HTTPException, Request, Response, httpx, uvicorn, _DEPS_AVAILABLE
    if _DEPS_AVAILABLE:
        return
    try:
        fastapi_mod = importlib.import_module("fastapi")
        fastapi_responses = importlib.import_module("fastapi.responses")
        FastAPI = fastapi_mod.FastAPI
        HTTPException = fastapi_mod.HTTPException
        Request = fastapi_mod.Request
        Response = fastapi_responses.Response
        httpx = importlib.import_module("httpx")
        uvicorn = importlib.import_module("uvicorn")
        _DEPS_AVAILABLE = True
    except Exception:  # pragma: no cover - optional deps
        FastAPI = None
        HTTPException = None
        Request = None
        Response = None
        httpx = None
        uvicorn = None
        _DEPS_AVAILABLE = False


def _assert_deps() -> None:
    _load_deps()
    if not _DEPS_AVAILABLE:
        raise RuntimeError(
            "Required dependencies not available (fastapi, uvicorn, httpx). "
            "Install with: pip install fastapi uvicorn httpx"
        )


def _resolve_upstream_key() -> str | None:
    return (
        os.environ.get("TS_STREAMLIT_API_KEY")
        or os.environ.get("TS_OPENAI_STREAMLIT")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("TREND_LLM_API_KEY")
    )


def _filter_response_headers(headers: dict[str, Any]) -> dict[str, str]:
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-encoding",
        "content-length",
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}


class LLMProxy:
    """OpenAI-compatible proxy that keeps API keys server-side."""

    def __init__(self, upstream_base: str | None = None) -> None:
        _assert_deps()
        if FastAPI is None or httpx is None:  # pragma: no cover
            raise RuntimeError("FastAPI/httpx dependencies are required for LLMProxy")
        self.upstream_base = (
            upstream_base or os.environ.get("TS_LLM_PROXY_UPSTREAM") or "https://api.openai.com"
        ).rstrip("/")
        self.auth_token = os.environ.get("TS_LLM_PROXY_TOKEN")
        self.app = FastAPI(title="LLM Proxy", version="1.0.0")
        self.client = httpx.AsyncClient(timeout=90)
        self._register_routes()

    def _register_routes(self) -> None:
        assert self.app is not None

        @self.app.get("/health")  # type: ignore[misc]
        async def _health() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.api_route(  # type: ignore[misc]
            "/v1/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        async def _forward(path: str, request: Any) -> Any:
            return await self._handle_request(request, path)

    async def _handle_request(self, request: Any, path: str) -> Any:
        assert Response is not None and HTTPException is not None and httpx is not None
        if self.auth_token:
            auth_header = request.headers.get("authorization") or ""
            if not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing proxy token")
            token = auth_header.removeprefix("Bearer ").strip()
            if not secrets.compare_digest(token, self.auth_token):
                raise HTTPException(status_code=403, detail="Invalid proxy token")

        upstream_key = _resolve_upstream_key()
        if not upstream_key:
            raise HTTPException(status_code=500, detail="Upstream API key not configured")

        normalized = path.lstrip("/")
        target_url = urljoin(f"{self.upstream_base}/", f"v1/{normalized}")
        raw_query = getattr(request.url, "query", "")
        query = (
            raw_query.decode("utf-8", errors="ignore")
            if isinstance(raw_query, bytes)
            else str(raw_query)
        )
        if query:
            target_url += f"?{query}"

        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers["authorization"] = f"Bearer {upstream_key}"
        body = await request.body()

        try:
            response = await self.client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            )
        except httpx.TimeoutException as exc:
            logger.warning("Upstream request timed out: %s", exc)
            raise HTTPException(status_code=504, detail="Upstream service timed out") from exc
        except httpx.RequestError as exc:
            logger.warning("Network error while contacting upstream: %s", exc)
            raise HTTPException(
                status_code=502, detail="Error contacting upstream service"
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("HTTP error from upstream: %s", exc)
            raise HTTPException(status_code=502, detail="Upstream service error") from exc

        filtered = _filter_response_headers(dict(response.headers))
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=filtered,
        )

    async def start(self, host: str = "0.0.0.0", port: int = 8799) -> None:  # noqa: D401
        _assert_deps()
        if uvicorn is None:
            raise RuntimeError("uvicorn dependency is required to start the proxy server")
        config = uvicorn.Config(app=self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        logger.info("Starting LLM proxy on %s:%s", host, port)
        logger.info("Forwarding to upstream at %s", self.upstream_base)
        await server.serve()

    async def close(self) -> None:  # noqa: D401
        _assert_deps()
        await self.client.aclose()


def run_proxy(
    upstream_base: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8799,
) -> None:
    """Run the LLM proxy synchronously (convenience wrapper)."""

    async def main() -> None:
        proxy = LLMProxy(upstream_base=upstream_base)
        try:
            await proxy.start(host=host, port=port)
        finally:
            await proxy.close()

    asyncio.run(main())
