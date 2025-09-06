"""WebSocket-capable proxy server for Streamlit applications.

This proxy forwards HTTP requests using httpx and WebSocket connections directly,
ensuring that Streamlit's real-time features work properly through the proxy.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urljoin

try:
    import httpx
    from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    from starlette.background import BackgroundTask
    import uvicorn
    import websockets
except ImportError:
    # Graceful degradation if dependencies aren't available
    httpx = None
    FastAPI = None
    Request = None
    Response = None
    WebSocket = None
    WebSocketDisconnect = None
    StreamingResponse = None
    BackgroundTask = None
    uvicorn = None
    websockets = None

logger = logging.getLogger(__name__)


class StreamlitProxy:
    """A proxy server that forwards both HTTP and WebSocket traffic to Streamlit.

    This solves the issue where Streamlit's frontend requires WebSocket endpoints
    like `/_stcore/stream` for bidirectional updates that aren't supported by
    simple HTTP-only proxies.
    """

    def __init__(self, streamlit_host: str = "localhost", streamlit_port: int = 8501):
        """Initialize the Streamlit proxy.

        Args:
            streamlit_host: Host where Streamlit is running
            streamlit_port: Port where Streamlit is running
        """
        if not all([httpx, FastAPI, uvicorn, websockets]):
            raise ImportError(
                "Required dependencies not available. Install with: "
                "pip install fastapi uvicorn httpx websockets"
            )

        self.streamlit_host = streamlit_host
        self.streamlit_port = streamlit_port
        self.streamlit_base_url = f"http://{streamlit_host}:{streamlit_port}"
        self.streamlit_ws_url = f"ws://{streamlit_host}:{streamlit_port}"

        self.app = FastAPI(title="Streamlit Proxy", version="1.0.0")
        self.client = httpx.AsyncClient()

        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up HTTP and WebSocket routes."""

        @self.app.websocket("/{path:path}")
        async def websocket_proxy(websocket: WebSocket, path: str):
            """Proxy WebSocket connections to Streamlit."""
            await self._handle_websocket(websocket, path)

        @self.app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        )
        async def http_proxy(request: Request, path: str):
            """Proxy HTTP requests to Streamlit."""
            return await self._handle_http_request(request, path)

    async def _handle_websocket(self, websocket: WebSocket, path: str) -> None:
        """Handle WebSocket connection forwarding to Streamlit.

        Args:
            websocket: The incoming WebSocket connection
            path: The WebSocket path being requested
        """
        await websocket.accept()

        # Construct the target WebSocket URL
        target_url = f"{self.streamlit_ws_url}/{path}"
        query_string = websocket.url.query
        if query_string:
            target_url += f"?{query_string}"

        logger.info(f"Proxying WebSocket connection to: {target_url}")

        try:
            async with websockets.connect(target_url) as target_ws:
                # Set up bidirectional forwarding
                async def forward_to_target():
                    """Forward messages from client to Streamlit."""
                    try:
                        while True:
                            message = await websocket.receive()
                            if "bytes" in message and message["bytes"] is not None:
                                await target_ws.send(message["bytes"])
                            elif "text" in message and message["text"] is not None:
                                await target_ws.send(message["text"])
                    except WebSocketDisconnect:
                        pass
                    except Exception as e:
                        logger.error(f"Error forwarding to target: {e}")

                async def forward_to_client():
                    """Forward messages from Streamlit to client."""
                    try:
                        async for message in target_ws:
                            if isinstance(message, bytes):
                                await websocket.send_bytes(message)
                            else:
                                await websocket.send_text(message)
                    except Exception as e:
                        logger.error(f"Error forwarding to client: {e}")

                # Run both forwarding tasks concurrently
                await asyncio.gather(
                    forward_to_target(), forward_to_client(), return_exceptions=True
                )

        except Exception as e:
            logger.error(f"WebSocket proxy error: {e}")
            await websocket.close(code=1011)

    async def _handle_http_request(self, request: Request, path: str) -> Response:
        """Handle HTTP request forwarding to Streamlit.

        Args:
            request: The incoming HTTP request
            path: The HTTP path being requested

        Returns:
            The proxied response from Streamlit
        """
        # Construct the target URL
        normalized_path = path if path.startswith("/") else f"/{path}"
        target_url = urljoin(self.streamlit_base_url, normalized_path)
        query_string = str(request.url.query)
        if query_string:
            target_url += f"?{query_string}"

        logger.debug(f"Proxying HTTP {request.method} request to: {target_url}")

        # Forward headers (excluding host)
        headers = dict(request.headers)
        headers.pop("host", None)

        try:
            # Read request body
            body = await request.body()

            # Forward the request to Streamlit
            response = await self.client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                follow_redirects=True,
            )

            # Prepare response headers
            response_headers = dict(response.headers)
            # Remove headers that should not be forwarded
            for header in ["content-encoding", "content-length", "transfer-encoding"]:
                response_headers.pop(header, None)

            # Return streaming response for large content
            async def generate():
                async for chunk in response.aiter_bytes():
                    yield chunk

            return StreamingResponse(
                generate(),
                status_code=response.status_code,
                headers=response_headers,
                background=BackgroundTask(response.aclose),
            )

        except httpx.RequestError as e:
            logger.error(f"HTTP proxy error: {e}")
            return Response(
                content=f"Proxy error: {str(e)}",
                status_code=502,
                media_type="text/plain",
            )

    async def start(self, host: str = "0.0.0.0", port: int = 8500) -> None:
        """Start the proxy server.

        Args:
            host: Host to bind the proxy server to
            port: Port to bind the proxy server to
        """
        config = uvicorn.Config(app=self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        logger.info(f"Starting Streamlit proxy on {host}:{port}")
        logger.info(f"Forwarding to Streamlit at {self.streamlit_base_url}")
        await server.serve()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


def run_proxy(
    streamlit_host: str = "localhost",
    streamlit_port: int = 8501,
    proxy_host: str = "0.0.0.0",
    proxy_port: int = 8500,
) -> None:
    """Run the Streamlit proxy server.

    Args:
        streamlit_host: Host where Streamlit is running
        streamlit_port: Port where Streamlit is running
        proxy_host: Host to bind the proxy server to
        proxy_port: Port to bind the proxy server to
    """

    async def main():
        proxy = StreamlitProxy(streamlit_host, streamlit_port)
        try:
            await proxy.start(proxy_host, proxy_port)
        finally:
            await proxy.close()

    asyncio.run(main())


if __name__ == "__main__":
    run_proxy()
