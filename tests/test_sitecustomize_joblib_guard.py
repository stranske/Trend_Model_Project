"""Tests covering the opt-in ``trend_model._sitecustomize`` helpers."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from trend_model import _sitecustomize as sitecustom

import sitecustomize as sitecustom_shim

SRC_PATH = str(sitecustom.SRC_DIR)
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FLAG = sitecustom_shim.ENV_FLAG


@pytest.fixture(autouse=True)
def restore_sys_path() -> Iterator[None]:
    original = list(sys.path)
    yield
    sys.path[:] = original


def test_maybe_apply_is_noop_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sitecustom.ENV_FLAG, raising=False)

    def boom(*args, **kwargs):  # pragma: no cover - executed only on failure
        raise AssertionError("joblib resolution should not occur when flag disabled")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    baseline = [entry for entry in sys.path if entry != SRC_PATH]
    monkeypatch.setattr(sys, "path", list(baseline))

    sitecustom.maybe_apply()

    assert sys.path == baseline


def test_apply_raises_when_joblib_points_inside_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_spec = SimpleNamespace(origin=str(REPO_ROOT / "joblib.py"))
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):  # type: ignore[override]
        if name == "joblib":
            return stub_spec
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(ImportError):
        sitecustom.apply()


def test_apply_allows_repo_virtualenv_site_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = REPO_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "joblib"
    venv_spec = SimpleNamespace(origin=str(candidate / "__init__.py"))
    original_find_spec = importlib.util.find_spec

    def virtualenv_find_spec(name: str, *args, **kwargs):  # type: ignore[override]
        if name == "joblib":
            return venv_spec
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", virtualenv_find_spec)
    sitecustom.apply()


def test_maybe_apply_inserts_src_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sitecustom.ENV_FLAG, "1")
    monkeypatch.setattr(sys, "path", [])
    sitecustom.maybe_apply()

    assert sys.path[0] == SRC_PATH


def test_random_import_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sitecustom.ENV_FLAG, raising=False)
    sys.modules.pop("trend_model._sitecustomize", None)
    list(sys.path)

    __import__("random")

    assert "trend_model._sitecustomize" not in sys.modules


def test_sitecustomize_default_import_is_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading the shim without the flag must not import the bootstrap
    module."""

    monkeypatch.delenv(ENV_FLAG, raising=False)
    sys.modules.pop("trend_model._sitecustomize", None)

    importlib.reload(sitecustom_shim)

    assert "trend_model._sitecustomize" not in sys.modules


@pytest.mark.parametrize("flag_value", ["0", "", "true", "yes"])
def test_opt_in_requires_exact_flag(
    monkeypatch: pytest.MonkeyPatch, flag_value: str
) -> None:
    """Only the explicit opt-in value should trigger the bootstrap shim."""

    with monkeypatch.context() as ctx:
        ctx.setenv(ENV_FLAG, flag_value)
        sys.modules.pop("trend_model._sitecustomize", None)

        importlib.reload(sitecustom_shim)
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
