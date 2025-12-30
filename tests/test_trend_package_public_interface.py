"""Additional tests covering the lightweight :mod:`trend` package shim."""

from __future__ import annotations

import importlib

import pytest


def reload_trend():
    """Reload the ``trend`` package to ensure patched metadata is visible."""

    import trend

    return importlib.reload(trend)


def test_version_attribute_uses_metadata(monkeypatch) -> None:
    trend = reload_trend()

    monkeypatch.setattr(trend._metadata, "version", lambda _: "9.9.9")

    assert trend.__version__ == "9.9.9"


def test_version_attribute_fallback(monkeypatch) -> None:
    trend = reload_trend()

    class DummyError(Exception):
        pass

    monkeypatch.setattr(trend._metadata, "PackageNotFoundError", DummyError)

    def raise_error(_: str) -> str:
        raise DummyError

    monkeypatch.setattr(trend._metadata, "version", raise_error)

    assert trend.__version__ == "0.0.dev0"


def test_unknown_attribute_raises_attribute_error() -> None:
    trend = reload_trend()

    with pytest.raises(AttributeError):
        trend.unknown_attribute
