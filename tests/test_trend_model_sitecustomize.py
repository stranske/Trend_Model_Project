from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from trend_model import _sitecustomize as site


def test_maybe_apply_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(site.ENV_FLAG, raising=False)
    called = []
    monkeypatch.setattr(site, "apply", lambda: called.append(True))

    site.maybe_apply()
    assert called == []

    monkeypatch.setenv(site.ENV_FLAG, "1")
    site.maybe_apply()
    assert called == [True]


def test_bootstrap_and_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(site, "SRC_DIR", Path.cwd())
    monkeypatch.setattr(site.sys, "path", [])

    site.bootstrap()
    assert site.sys.path[0] == str(Path.cwd())

    called = []
    monkeypatch.setattr(site, "_ensure_src_on_sys_path", lambda: called.append("src"))
    monkeypatch.setattr(
        site, "_ensure_joblib_external", lambda: called.append("joblib")
    )

    site.apply()
    assert called == ["src", "joblib"]


def test_ensure_src_on_sys_path_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    temp_dir = Path.cwd()
    monkeypatch.setattr(site, "SRC_DIR", temp_dir)
    monkeypatch.setattr(site.sys, "path", ["existing"])

    site._ensure_src_on_sys_path()
    assert str(temp_dir) in site.sys.path

    # Second invocation should not duplicate entries.
    site._ensure_src_on_sys_path()
    assert site.sys.path.count(str(temp_dir)) == 1


def test_ensure_joblib_external_behaviour(monkeypatch: pytest.MonkeyPatch) -> None:
    base = Path.cwd()

    def fake_find_spec(name: str) -> SimpleNamespace | None:
        if name == "joblib":
            return SimpleNamespace(origin=str(base / "site-packages" / "joblib.py"))
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    # Acceptable when resolving from site-packages.
    site._ensure_joblib_external()

    # Missing module should be ignored gracefully.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    site._ensure_joblib_external()

    # Resolution within the repository should raise an error.
    repo_stub = SimpleNamespace(origin=str(site.REPO_ROOT / "joblib.py"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: repo_stub)
    with pytest.raises(ImportError, match="repository stub"):
        site._ensure_joblib_external()

    # Unexpected location without site-packages/dist-packages should raise.
    elsewhere = SimpleNamespace(origin=str(Path("/tmp/joblib.py")))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: elsewhere)
    with pytest.raises(ImportError, match="should resolve"):
        site._ensure_joblib_external()
