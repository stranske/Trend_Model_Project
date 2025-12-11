"""FastAPI health wrapper for trend portfolio app.

This module provides a FastAPI-based health check endpoint that can be
used for Docker health checks, load balancer health checks, and CI/CD
pipelines. It serves as a lightweight alternative to the full Streamlit
app for health monitoring.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imports for type hints only
    from fastapi import FastAPI as FastAPIType
    from fastapi.responses import PlainTextResponse as PlainTextResponseType
else:  # runtime fallbacks keep optional deps optional
    FastAPIType = Any  # type: ignore[assignment]
    PlainTextResponseType = Any  # type: ignore[assignment]

FastAPI: type[FastAPIType] | None
PlainTextResponse: type[PlainTextResponseType] | None

try:  # pragma: no cover - import side effect
    from fastapi import FastAPI as _FastAPI
    from fastapi.responses import PlainTextResponse as _PlainTextResponse
except (ImportError, ModuleNotFoundError):  # FastAPI missing
    FastAPI = None
    PlainTextResponse = None
else:
    FastAPI = _FastAPI
    PlainTextResponse = _PlainTextResponse

uvicorn: ModuleType | None
try:  # pragma: no cover - import side effect
    import uvicorn as _uvicorn
except (ImportError, ModuleNotFoundError):  # uvicorn missing
    uvicorn = None
else:
    uvicorn = _uvicorn


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

    async def _health_check() -> str:
        return "OK"

    async def _root_health_check() -> str:
        return "OK"

    app_obj.add_api_route(
        "/health",
        _health_check,
        methods=["GET"],
        response_class=PlainTextResponse,
    )
    app_obj.add_api_route(
        "/",
        _root_health_check,
        methods=["GET"],
        response_class=PlainTextResponse,
    )

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
