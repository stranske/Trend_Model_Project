"""Tests for the lightweight :mod:`trend` package shim."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
from collections.abc import Callable

import pytest


def _reload_trend(monkeypatch, version_callable: Callable[[str], str]) -> object:
    """Reload :mod:`trend` with a patched metadata lookup."""

    monkeypatch.delitem(sys.modules, "trend", raising=False)
    monkeypatch.setattr(metadata, "version", version_callable)

    module = importlib.import_module("trend")
    return importlib.reload(module)


def test_dunder_version_uses_package_metadata(monkeypatch):
    """``trend.__version__`` should proxy to the installed package metadata."""

    expected_version = "9.9.9"
    trend_module = _reload_trend(monkeypatch, lambda name: expected_version)

    assert trend_module.__version__ == expected_version


def test_dunder_version_falls_back_in_dev_mode(monkeypatch):
    """Missing package metadata should trigger the development fallback
    string."""

    def raise_missing(_name: str) -> str:  # pragma: no cover - defensive guard
        raise metadata.PackageNotFoundError

    trend_module = _reload_trend(monkeypatch, raise_missing)

    assert trend_module.__version__ == "0.0.dev0"


def test_unknown_attributes_raise_attribute_error(monkeypatch):
    """Any attribute other than ``__version__`` should raise
    ``AttributeError``."""

    trend_module = _reload_trend(monkeypatch, lambda name: "1.2.3")

    with pytest.raises(AttributeError):
        trend_module.missing
