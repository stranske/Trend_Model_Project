"""Tests for the FastAPI server."""

import asyncio
import runpy
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from trend_analysis import api_server
from trend_analysis.api_server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Trend Analysis API"
    assert data["version"] == "1.0.0"
    assert "docs" in data
    assert "health" in data


def test_lifespan_events(client):
    """Test that lifespan events are properly configured."""
    # The fact that the client can be created and used successfully
    # indicates that the lifespan context manager is working correctly
    response = client.get("/health")
    assert response.status_code == 200

    # Test that the app can handle multiple requests
    response2 = client.get("/")
    assert response2.status_code == 200


def test_api_docs_accessible(client):
    """Test that OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/redoc")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200


def test_api_rejects_risky_patch_without_confirmation(client):
    payload = {
        "config": {"portfolio": {"max_turnover": 1.0}},
        "patch": {
            "operations": [
                {"op": "remove", "path": "portfolio.max_turnover"},
            ],
            "risk_flags": [],
            "summary": "Remove turnover cap.",
        },
    }

    response = client.post("/config/patch", json=payload)

    assert response.status_code == 400
    assert "Risky changes detected" in response.json()["detail"]


def test_api_allows_risky_patch_with_confirmation(client):
    payload = {
        "config": {"portfolio": {"max_turnover": 1.0}},
        "patch": {
            "operations": [
                {"op": "remove", "path": "portfolio.max_turnover"},
            ],
            "risk_flags": [],
            "summary": "Remove turnover cap.",
        },
        "confirm_risky": True,
    }

    response = client.post("/config/patch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "portfolio" in body["config"]


def test_api_rejects_unknown_key_review_without_confirmation(client):
    payload = {
        "config": {"analysis": {"top_n": 10}},
        "patch": {
            "operations": [
                {"op": "set", "path": "analysis.top_n", "value": 12},
            ],
            "needs_review": True,
            "risk_flags": [],
            "summary": "Update selection count.",
        },
    }

    response = client.post("/config/patch", json=payload)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "UNKNOWN_KEYS" in detail
    assert "confirm_risky" in detail


def test_lifespan_context_logs_startup_and_shutdown(caplog):
    """Ensure the custom lifespan context logs both lifecycle events."""

    caplog.set_level("INFO")

    async def _run() -> None:
        async with api_server.lifespan(app):
            # Keep the context manager alive briefly so startup fires.
            await asyncio.sleep(0)

    asyncio.run(_run())

    assert "Starting up trend analysis API server" in caplog.text
    assert "Shutting down trend analysis API server" in caplog.text


def test_run_invokes_uvicorn(monkeypatch):
    """The ``run`` helper should delegate to ``uvicorn.run`` with arguments."""

    calls: list[dict[str, object]] = []

    module = SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "uvicorn", module)

    host, port = api_server.run(host="0.0.0.0", port=1234)

    assert host == "0.0.0.0"
    assert port == 1234
    assert calls == [
        (
            (app,),
            {
                "host": "0.0.0.0",
                "port": 1234,
                "reload": False,
                "log_level": "info",
            },
        )
    ]


def test_api_server_module_entrypoint(monkeypatch):
    """Running the module as ``python -m`` should invoke the ``run`` helper."""

    called: dict[str, tuple[str, int]] = {}

    def fake_run(host: str, port: int) -> None:
        called["args"] = (host, port)

    monkeypatch.setattr(api_server, "run", fake_run)

    runpy.run_module("trend_analysis.api_server.__main__", run_name="__main__")

    assert called["args"] == ("0.0.0.0", 8000)
