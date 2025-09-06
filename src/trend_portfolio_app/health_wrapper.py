"""FastAPI health wrapper for trend portfolio app.

This module provides a FastAPI-based health check endpoint that can be
used for Docker health checks, load balancer health checks, and CI/CD
pipelines. It serves as a lightweight alternative to the full Streamlit
app for health monitoring.
"""

from __future__ import annotations

import os
import sys

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse
except ImportError:
    # Graceful fallback if FastAPI/uvicorn are not installed
    FastAPI = None
    PlainTextResponse = None
    uvicorn = None


def create_app():
    """Create FastAPI application with health endpoint."""
    if FastAPI is None:
        raise ImportError(
            "FastAPI is required but not installed. "
            "Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Trend Portfolio Health Service",
        description="Health check service for trend portfolio application",
        version="1.0.0",
        docs_url=None,  # Disable docs for security
        redoc_url=None,  # Disable redoc for security
    )

    @app.get("/health", response_class=PlainTextResponse)
    async def health_check() -> str:
        """Health check endpoint that returns 'OK' when service is healthy."""
        return "OK"

    @app.get("/", response_class=PlainTextResponse)
    async def root_health_check() -> str:
        """Root endpoint that also serves as health check for compatibility."""
        return "OK"

    return app


# Create the app instance (only when FastAPI is available)
if FastAPI is not None:
    app = create_app()
else:
    app = None


def main() -> None:
    """Main entry point for running the health wrapper service."""
    if uvicorn is None:
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


if __name__ == "__main__":
    main()
