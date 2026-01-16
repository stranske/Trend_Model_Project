"""FastAPI server for trend analysis API.

Provides REST API endpoints for trend analysis operations with modern
lifespan context manager for startup/shutdown events.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from trend_analysis.tool_layer import ToolLayer

logger = logging.getLogger(__name__)
_tool_layer = None


def _get_tool_layer() -> "ToolLayer":
    global _tool_layer
    if _tool_layer is None:
        from trend_analysis.tool_layer import ToolLayer

        _tool_layer = ToolLayer()
    return _tool_layer


class ConfigPatchRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    patch: dict[str, Any] = Field(default_factory=dict)
    confirm_risky: bool = False


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


def _risky_patch_flags(patch: "ConfigPatch") -> list[str]:
    flags = [flag.value for flag in patch.risk_flags]
    if patch.needs_review:
        flags.append("UNKNOWN_KEYS")
    return flags


def _build_risky_detail(flags: list[str]) -> str:
    flags_text = ", ".join(flags)
    return f"Risky changes detected ({flags_text}). Set confirm_risky=True to apply."


async def _risky_change_guard(request: Request, call_next):
    if request.method not in {"POST", "PUT", "PATCH"}:
        return await call_next(request)

    body = await request.body()
    if not body:
        return await call_next(request)

    # Preserve the body for downstream request parsing.
    request._body = body  # type: ignore[attr-defined]

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return await call_next(request)

    if not isinstance(payload, dict):
        return await call_next(request)

    patch_payload = payload.get("patch")
    if not isinstance(patch_payload, dict):
        return await call_next(request)

    try:
        from trend_analysis.config.patch import ConfigPatch

        patch_obj = ConfigPatch.model_validate(patch_payload)
    except Exception:
        return await call_next(request)

    confirm_risky = payload.get("confirm_risky") is True
    if not confirm_risky:
        flags = _risky_patch_flags(patch_obj)
        if flags:
            detail = _build_risky_detail(flags)
            return JSONResponse(status_code=400, content={"detail": detail})

    return await call_next(request)


app.middleware("http")(_risky_change_guard)


async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Trend Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


app.add_api_route("/health", health_check, methods=["GET"])
app.add_api_route("/", root, methods=["GET"])


@app.post("/config/patch")
async def apply_config_patch(payload: ConfigPatchRequest) -> dict[str, Any]:
    """Apply a config patch with risk confirmation enforcement."""
    tool = _get_tool_layer()
    result = tool.apply_patch(
        payload.config,
        payload.patch,
        confirm_risky=payload.confirm_risky,
    )
    if result.status != "success":
        raise HTTPException(status_code=400, detail=result.message or "Invalid config patch.")
    return {"status": "success", "config": result.data}


@app.post("/config/patch/preview")
async def preview_config_patch(payload: ConfigPatchRequest) -> dict[str, Any]:
    """Preview a config patch with risk confirmation enforcement."""
    tool = _get_tool_layer()
    result = tool.apply_patch(
        payload.config,
        payload.patch,
        confirm_risky=payload.confirm_risky,
    )
    if result.status != "success":
        raise HTTPException(status_code=400, detail=result.message or "Invalid config patch.")

    from trend_analysis.config.patch import diff_configs

    updated = result.data if isinstance(result.data, dict) else {}
    diff = diff_configs(dict(payload.config), updated)
    return {"status": "success", "config": updated, "diff": diff}


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
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
    return host, port


if __name__ == "__main__":
    run()
