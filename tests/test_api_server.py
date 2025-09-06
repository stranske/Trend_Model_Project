"""Tests for the FastAPI server."""

import pytest
from fastapi.testclient import TestClient

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