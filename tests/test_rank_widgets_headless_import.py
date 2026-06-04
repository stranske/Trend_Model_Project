from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_rank_widgets_imports_without_ipywidgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "trend_analysis.ui.rank_widgets", raising=False)
    monkeypatch.delitem(sys.modules, "ipywidgets", raising=False)

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "ipywidgets" or name.startswith("ipywidgets."):
            raise ImportError("ipywidgets deliberately hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = importlib.import_module("trend_analysis.ui.rank_widgets")

    assert module.__name__ == "trend_analysis.ui.rank_widgets"


def test_build_ui_still_requires_ipywidgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "trend_analysis.ui.rank_widgets", raising=False)
    monkeypatch.delitem(sys.modules, "ipywidgets", raising=False)

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "ipywidgets" or name.startswith("ipywidgets."):
            raise ImportError("ipywidgets deliberately hidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = importlib.import_module("trend_analysis.ui.rank_widgets")

    with pytest.raises(ImportError, match="ipywidgets is required"):
        module.build_ui()
