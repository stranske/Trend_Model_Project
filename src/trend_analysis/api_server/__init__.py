"""FastAPI server for trend analysis API.

Provides REST API endpoints for trend analysis operations with modern
lifespan context manager for startup/shutdown events.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Tuple

from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events.

    Modern FastAPI approach using context manager instead of deprecated
    @app.on_event() decorators.
    """
    # Startup logic
    logger.info("Starting up trend analysis API server")
    # Initialize any resources here (database connections, ML models, etc.)

    yield  # Application is running

    # Shutdown logic
    logger.info("Shutting down trend analysis API server")
    # Clean up resources here


# Create FastAPI app with lifespan context manager
app = FastAPI(
    title="Trend Analysis API",
    description="REST API for volatility-adjusted trend portfolio analysis",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Trend Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


def run(host: str = "127.0.0.1", port: int = 8000) -> Tuple[str, int]:
    """Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to

    Returns:
        Tuple of (host, port) for backward compatibility
    """
    import uvicorn

    uvicorn.run(
        "trend_analysis.api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
    return host, port


if __name__ == "__main__":
    run()
