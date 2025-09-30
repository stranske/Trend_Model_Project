"""Tests for the joblib guard enforced during interpreter bootstrap."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

import sitecustomize

REPO_ROOT = Path(__file__).resolve().parents[1]
FLAG = "TREND_MODEL_SITE_CUSTOMIZE"


def _reload_with_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload the shim after opting in to the bootstrap hooks."""

    monkeypatch.setenv(FLAG, "1")
    sys.modules.pop("trend_model._sitecustomize", None)
    importlib.reload(sitecustomize)


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
            _reload_with_opt_in(ctx)

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
        _reload_with_opt_in(ctx)

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
        _reload_with_opt_in(ctx)

    importlib.reload(sitecustomize)


def test_importing_project_module_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing project modules must not trigger the bootstrap by default."""

    monkeypatch.delenv(FLAG, raising=False)
    sys.modules.pop("trend_model._sitecustomize", None)

    import trend_analysis.pipeline  # noqa: F401  (arbitrary project module)

    assert "trend_model._sitecustomize" not in sys.modules


@pytest.mark.parametrize("flag_value", ["0", "", "true", "yes"])
def test_opt_in_requires_exact_flag(monkeypatch: pytest.MonkeyPatch, flag_value: str) -> None:
    """Only the explicit opt-in value should trigger the bootstrap shim."""

    with monkeypatch.context() as ctx:
        ctx.setenv(FLAG, flag_value)
        sys.modules.pop("trend_model._sitecustomize", None)

        importlib.reload(sitecustomize)
        assert "trend_model._sitecustomize" not in sys.modules
def test_bootstrap_inserts_src_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in bootstrap should prepend src/ exactly once."""

    module = importlib.import_module("trend_model._sitecustomize")
    module = importlib.reload(module)
    src_path = str(REPO_ROOT / "src")

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "path", [p for p in sys.path if p != src_path])

        module.bootstrap()
        assert sys.path[0] == src_path

        module.bootstrap()
        assert sys.path.count(src_path) == 1


def test_bootstrap_noop_when_src_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bootstrap should not mutate sys.path if src/ is absent."""

    module = importlib.import_module("trend_model._sitecustomize")
    module = importlib.reload(module)
    fake_src = REPO_ROOT / "_does_not_exist_src"

    with monkeypatch.context() as ctx:
        ctx.setattr(module, "SRC_DIR", fake_src)
        ctx.setattr(sys, "path", [])

        module.bootstrap()

        assert sys.path == []
