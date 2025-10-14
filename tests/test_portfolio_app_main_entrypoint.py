"""Tests for the ``trend_portfolio_app.__main__`` module."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


def _prepare_streamlit_cli() -> None:
    """Register stub modules so the entrypoint can import streamlit CLI."""

    streamlit_module = ModuleType("streamlit")
    streamlit_web = ModuleType("streamlit.web")
    streamlit_cli = ModuleType("streamlit.web.cli")
    streamlit_cli.main = lambda: None  # type: ignore[attr-defined]

    sys.modules["streamlit"] = streamlit_module
    sys.modules["streamlit.web"] = streamlit_web
    sys.modules["streamlit.web.cli"] = streamlit_cli


def test_main_adds_src_path_and_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_streamlit_cli()
    module = importlib.import_module("trend_portfolio_app.__main__")

    captured = {}

    def fake_main() -> None:
        captured["called"] = True

    repo_root = module.Path(module.__file__).resolve().parent.parent.parent
    src_path = str(repo_root / "src")

    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys.modules["streamlit.web.cli"], "main", fake_main)

    module.main()

    assert captured["called"] is True
    assert sys.argv[:2] == ["streamlit", "run"]
    assert sys.argv[2].endswith("trend_portfolio_app/app.py")
    assert sys.path[0] == src_path
