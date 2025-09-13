"""FastAPI health wrapper for trend portfolio app.

This module provides a FastAPI-based health check endpoint that can be
used for Docker health checks, load balancer health checks, and CI/CD
pipelines. It serves as a lightweight alternative to the full Streamlit
app for health monitoring.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Optional dependency sentinels (exposed for tests to monkeypatch)
try:  # pragma: no cover - import side effect
    from fastapi import FastAPI  # type: ignore
    from fastapi.responses import PlainTextResponse  # type: ignore
except (ImportError, ModuleNotFoundError):  # FastAPI missing
    FastAPI = None  # type: ignore
    PlainTextResponse = None  # type: ignore

try:  # pragma: no cover - import side effect
    import uvicorn  # type: ignore
except (ImportError, ModuleNotFoundError):  # uvicorn missing
    uvicorn = None  # type: ignore


def create_app() -> Any:
    """Create FastAPI application with health endpoints.

    Returns a FastAPI app if dependencies are available, otherwise
    raises ImportError.
    """
    if FastAPI is None or PlainTextResponse is None:  # pragma: no cover - defensive
        raise ImportError(
            "FastAPI is required but not installed. Install with: pip install fastapi uvicorn"
        )

    app_obj = FastAPI(
        title="Trend Portfolio Health Service",
        description="Health check service for trend portfolio application",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )

    @app_obj.get("/health", response_class=PlainTextResponse)
    async def health_check() -> str:  # noqa: D401
        return "OK"

    @app_obj.get("/", response_class=PlainTextResponse)
    async def root_health_check() -> str:  # noqa: D401
        return "OK"

    return app_obj


if FastAPI is not None:  # Create app eagerly when possible
    try:
        app = create_app()
    except (
        ImportError,
        AttributeError,
        TypeError,
    ) as e:  # pragma: no cover - defensive
        print(f"Warning: Failed to create FastAPI app: {e}", file=sys.stderr)
        app = None
else:
    app = None


def main() -> None:
    """Main entry point for running the health wrapper service."""
    if uvicorn is None:  # pragma: no cover
        print("ERROR: uvicorn is required but not installed.", file=sys.stderr)
        print("Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    # Configuration from environment variables
    host = os.environ.get("HEALTH_HOST", "0.0.0.0")
    port = int(os.environ.get("HEALTH_PORT", "8000"))

    print(f"Starting health wrapper service on {host}:{port}")

    # Use fully qualified module name to avoid import errors
    uvicorn.run(
        "trend_portfolio_app.health_wrapper:app",
        host=host,
        port=port,
        reload=False,  # Disable reload for production
        access_log=False,  # Minimal logging for health service
    )


if __name__ == "__main__":  # pragma: no cover
    main()
