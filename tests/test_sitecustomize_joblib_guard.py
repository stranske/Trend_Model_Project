"""Tests for the joblib guard enforced during interpreter bootstrap."""

from __future__ import annotations

import importlib
import importlib.util
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


def test_sitecustomize_allows_repo_virtualenv_site_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A virtualenv within the repository should be treated as third-party."""

    venv_spec = types.SimpleNamespace(
        origin=str(
            REPO_ROOT
            / ".venv"
            / "lib"
            / "python3.11"
            / "site-packages"
            / "joblib"
            / "__init__.py"
        )
    )
    original_find_spec = importlib.util.find_spec

    def virtualenv_find_spec(name: str, *args, **kwargs):  # type: ignore[override]
        if name == "joblib":
            return venv_spec
        return original_find_spec(name, *args, **kwargs)

    with monkeypatch.context() as ctx:
        ctx.setattr(importlib.util, "find_spec", virtualenv_find_spec)
        importlib.reload(sitecustomize)

    importlib.reload(sitecustomize)


def test_sitecustomize_allows_missing_joblib(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard should defer to import-time error when joblib is absent."""

    original_find_spec = importlib.util.find_spec

    def missing_find_spec(name: str, *args, **kwargs):  # type: ignore[override]
        if name == "joblib":
            return None
        return original_find_spec(name, *args, **kwargs)

    with monkeypatch.context() as ctx:
        ctx.setattr(importlib.util, "find_spec", missing_find_spec)
        # No exception is expected because the guard should defer to the
        # subsequent import-time ModuleNotFoundError.
        importlib.reload(sitecustomize)

    importlib.reload(sitecustomize)
