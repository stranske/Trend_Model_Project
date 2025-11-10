"""Coverage for the lightweight ``trend`` namespace package."""

from __future__ import annotations

import importlib

import pytest

import trend


def test_dunder_getattr_returns_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trend._metadata, "version", lambda name: "1.2.3")
    assert trend.__getattr__("__version__") == "1.2.3"


def test_dunder_getattr_dev_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_version(name: str) -> str:
        raise trend._metadata.PackageNotFoundError

    monkeypatch.setattr(trend._metadata, "version", fake_version)
    assert trend.__getattr__("__version__") == "0.0.dev0"


def test_dunder_getattr_unknown_attribute() -> None:
    with pytest.raises(AttributeError):
        trend.__getattr__("unknown")


def test_module_reload_preserves_dunder_getattr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trend._metadata, "version", lambda name: "2.0.0")
    reloaded = importlib.reload(trend)
    assert reloaded.__getattr__("__version__") == "2.0.0"
