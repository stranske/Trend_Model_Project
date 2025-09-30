"""Tests covering the opt-in ``trend_model._sitecustomize`` helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from trend_model import _sitecustomize as sitecustom

SRC_PATH = str(sitecustom.SRC_DIR)
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    sitecustom.maybe_apply()
    assert SRC_PATH not in sys.path


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
    if SRC_PATH in sys.path:
        sys.path.remove(SRC_PATH)

    sitecustom.maybe_apply()

    assert sys.path[0] == SRC_PATH


def test_random_import_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sitecustom.ENV_FLAG, raising=False)
    baseline = list(sys.path)

    __import__("random")

    assert sys.path == baseline
