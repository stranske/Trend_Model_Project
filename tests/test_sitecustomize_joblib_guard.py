"""Tests for the joblib guard enforced during interpreter bootstrap."""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

import sitecustomize

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sitecustomize_raises_when_joblib_points_inside_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading with a repository-local joblib should raise ImportError."""

    stub_spec = types.SimpleNamespace(origin=str(REPO_ROOT / "joblib.py"))
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):  # type: ignore[override]
        if name == "joblib":
            return stub_spec
        return original_find_spec(name, *args, **kwargs)

    with monkeypatch.context() as ctx:
        ctx.setattr(importlib.util, "find_spec", fake_find_spec)
        with pytest.raises(ImportError):
            importlib.reload(sitecustomize)

    importlib.reload(sitecustomize)
