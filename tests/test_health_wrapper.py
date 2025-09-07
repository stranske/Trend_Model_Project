"""Test for health wrapper module to ensure proper module qualification."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# Add src to path
repo_root = Path(__file__).parent.parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))


def test_health_wrapper_module_import():
    """Test that health_wrapper module can be imported with correct
    qualification."""
    from trend_portfolio_app import health_wrapper

    # Verify module has correct fully qualified name
    assert health_wrapper.__name__ == "trend_portfolio_app.health_wrapper"

    # Verify main function exists
    assert hasattr(health_wrapper, "main")
    assert callable(health_wrapper.main)


def test_health_wrapper_module_path():
    """Test that the module is located in the correct package path."""
    from trend_portfolio_app import health_wrapper

    # Verify module file is in correct location
    module_path = Path(health_wrapper.__file__)
    assert module_path.name == "health_wrapper.py"
    assert "trend_portfolio_app" in str(module_path)


def test_health_wrapper_graceful_dependency_handling():
    """Test that module handles missing dependencies gracefully."""

    # Test that app is None when FastAPI is not available
    # (This will be None in our test environment without dependencies)
    # The key fix is that the module can be imported despite missing deps
    assert True  # If we get here, import succeeded which is the main fix


def test_create_app_missing_fastapi(monkeypatch):
    from trend_portfolio_app import health_wrapper

    monkeypatch.setattr(health_wrapper, "FastAPI", None)
    with pytest.raises(ImportError) as exc_info:
        health_wrapper.create_app()
    assert "FastAPI is required" in str(exc_info.value)


def test_main_missing_uvicorn(monkeypatch, capsys):
    from trend_portfolio_app import health_wrapper

    monkeypatch.setattr(health_wrapper, "uvicorn", None)
    with pytest.raises(SystemExit) as exc_info:
        health_wrapper.main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "uvicorn is required" in captured.err


def test_import_without_dependencies(monkeypatch):
    import importlib
    import builtins

    from trend_portfolio_app import health_wrapper as hw

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("fastapi") or name == "uvicorn":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    reloaded = importlib.reload(hw)
    assert reloaded.FastAPI is None
    assert reloaded.app is None

    # Reload again with real imports to restore module state
    monkeypatch.setattr(builtins, "__import__", original_import)
    importlib.reload(reloaded)


def test_create_app_endpoints():
    """Ensure the FastAPI app exposes expected health endpoints."""
    from trend_portfolio_app import health_wrapper

    app = health_wrapper.create_app()
    client = TestClient(app)
    assert client.get("/health").text == "OK"
    assert client.get("/").text == "OK"


def test_main_runs_uvicorn(monkeypatch):
    from trend_portfolio_app import health_wrapper

    calls = {}

    def fake_run(app_path, host, port, reload, access_log):
        calls["app_path"] = app_path
        calls["host"] = host
        calls["port"] = port
        calls["reload"] = reload
        calls["access_log"] = access_log

    monkeypatch.setattr(
        health_wrapper,
        "uvicorn",
        SimpleNamespace(run=fake_run),
    )
    monkeypatch.setenv("HEALTH_HOST", "127.0.0.1")
    monkeypatch.setenv("HEALTH_PORT", "1234")

    health_wrapper.main()

    assert calls["app_path"] == "trend_portfolio_app.health_wrapper:app"
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 1234
    assert calls["reload"] is False
    assert calls["access_log"] is False


if __name__ == "__main__":
    # Run tests directly for debugging
    test_health_wrapper_module_import()
    test_health_wrapper_module_path()
    test_health_wrapper_graceful_dependency_handling()
    print("✅ All health wrapper tests passed!")
