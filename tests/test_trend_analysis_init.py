"""Tests covering the public package initializer."""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest

import trend_analysis as ta


def test_version_falls_back_when_metadata_missing(monkeypatch, request):
    def raise_missing(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError()

    monkeypatch.setattr(importlib.metadata, "version", raise_missing)
    importlib.reload(ta)
    assert ta.__version__ == "0.1.0-dev"

    request.addfinalizer(lambda: importlib.reload(ta))


def test_getattr_raises_for_unknown_attribute():
    with pytest.raises(AttributeError):
        getattr(ta, "does_not_exist")
