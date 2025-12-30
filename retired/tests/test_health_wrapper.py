import pytest
from trend_portfolio_app import health_wrapper


@pytest.mark.skipif(health_wrapper.FastAPI is None, reason="FastAPI not installed")
def test_create_app_registers_health_routes():
    app = health_wrapper.create_app()

    testclient_mod = pytest.importorskip(
        "fastapi.testclient", reason="FastAPI test client missing"
    )
    TestClient = testclient_mod.TestClient

    client = TestClient(app)
    for path in ["/", "/health"]:
        response = client.get(path)
        assert response.status_code == 200
        assert response.text == "OK"


def test_create_app_requires_fastapi(monkeypatch):
    monkeypatch.setattr(health_wrapper, "FastAPI", None)
    monkeypatch.setattr(health_wrapper, "PlainTextResponse", None)

    with pytest.raises(ImportError):
        health_wrapper.create_app()


@pytest.mark.skipif(health_wrapper.FastAPI is None, reason="FastAPI not installed")
def test_main_exits_when_uvicorn_missing(monkeypatch, capsys):
    monkeypatch.setattr(health_wrapper, "uvicorn", None)

    with pytest.raises(SystemExit) as excinfo:
        health_wrapper.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "uvicorn is required" in captured.err
    assert "pip install uvicorn" in captured.err


@pytest.mark.skipif(
    health_wrapper.FastAPI is None or health_wrapper.uvicorn is None,
    reason="FastAPI or uvicorn not installed",
)
def test_main_runs_uvicorn_with_env_overrides(monkeypatch, capsys):
    calls = []

    def fake_run(app_path, **kwargs):
        calls.append((app_path, kwargs))

    monkeypatch.setenv("HEALTH_HOST", "127.0.0.1")
    monkeypatch.setenv("HEALTH_PORT", "1234")
    monkeypatch.setattr(
        health_wrapper, "uvicorn", type("UV", (), {"run": staticmethod(fake_run)})
    )

    health_wrapper.main()

    captured = capsys.readouterr()
    assert "Starting health wrapper service on 127.0.0.1:1234" in captured.out
    assert calls == [
        (
            "trend_portfolio_app.health_wrapper:app",
            {
                "host": "127.0.0.1",
                "port": 1234,
                "reload": False,
                "access_log": False,
            },
        )
    ]
