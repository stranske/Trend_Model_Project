"""Regression tests for the ``trend_portfolio_app.__main__`` module."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Iterable

import pytest


def _install_streamlit_stub(monkeypatch: pytest.MonkeyPatch, calls: list[Iterable[str]]) -> None:
    """Register a lightweight ``streamlit.web.cli`` stub in ``sys.modules``."""

    def fake_main() -> None:
        calls.append(tuple(sys.argv))

    streamlit_pkg = types.ModuleType("streamlit")
    web_pkg = types.ModuleType("streamlit.web")
    cli_module = types.ModuleType("streamlit.web.cli")
    cli_module.main = fake_main  # type: ignore[attr-defined]
    streamlit_pkg.web = web_pkg  # type: ignore[attr-defined]
    web_pkg.cli = cli_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_pkg)
    monkeypatch.setitem(sys.modules, "streamlit.web", web_pkg)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", cli_module)


def test_main_injects_src_path_and_invokes_streamlit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = importlib.import_module("trend_portfolio_app.__main__")
    src_path = Path(module.__file__).resolve().parents[2] / "src"

    calls: list[Iterable[str]] = []
    _install_streamlit_stub(monkeypatch, calls)

    monkeypatch.setattr(sys, "argv", ["python", "-m", "trend_portfolio_app"])
    monkeypatch.setattr(sys, "path", ["/tmp/other"])

    module.main()

    captured = capsys.readouterr().out
    assert "Starting Streamlit app" in captured
    assert "health service" in captured

    assert calls == [("streamlit", "run", str(Path(module.__file__).resolve().parent / "app.py"))]
    assert sys.path[0] == str(src_path)


def test_main_avoids_duplicate_src_in_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("trend_portfolio_app.__main__")
    src_path = str(Path(module.__file__).resolve().parents[2] / "src")

    calls: list[Iterable[str]] = []
    _install_streamlit_stub(monkeypatch, calls)

    original_path = [src_path, "/tmp/placeholder"]
    monkeypatch.setattr(sys, "path", original_path[:])

    monkeypatch.setattr(sys, "argv", ["python", "-m", "trend_portfolio_app", "--arg"])

    module.main()

    assert sys.path == original_path
    assert calls == [("streamlit", "run", str(Path(module.__file__).resolve().parent / "app.py"))]


def test_module_entry_point_executes_main(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("trend_portfolio_app.__main__")

    calls: list[Iterable[str]] = []
    _install_streamlit_stub(monkeypatch, calls)

    src_path = str(Path(module.__file__).resolve().parents[2] / "src")
    monkeypatch.setattr(sys, "path", [src_path])
    monkeypatch.setattr(sys, "argv", ["python", "-m", "trend_portfolio_app"])

    runpy = importlib.import_module("runpy")
    runpy.run_module("trend_portfolio_app.__main__", run_name="__main__")

    assert calls == [("streamlit", "run", str(Path(module.__file__).resolve().parent / "app.py"))]
