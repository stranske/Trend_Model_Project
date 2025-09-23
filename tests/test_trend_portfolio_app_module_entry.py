"""Coverage extensions for ``trend_portfolio_app.__main__`` entrypoint."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _reload_module() -> ModuleType:
    module = importlib.import_module("trend_portfolio_app.__main__")
    return importlib.reload(module)


def test_main_inserts_src_path_and_invokes_streamlit(monkeypatch) -> None:
    module = _reload_module()
    expected_src = Path(module.__file__).resolve().parents[2] / "src"

    calls: list[str] = []
    cli_module = ModuleType("streamlit.web.cli")
    cli_module.main = lambda: calls.append("run")
    web_module = ModuleType("streamlit.web")
    web_module.cli = cli_module
    streamlit_module = ModuleType("streamlit")
    streamlit_module.web = web_module
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setitem(sys.modules, "streamlit.web", web_module)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", cli_module)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    monkeypatch.setattr(module.sys, "argv", ["python"])
    monkeypatch.setattr(module.sys, "path", [])

    module.main()

    assert module.sys.path[0] == str(expected_src)
    assert module.sys.argv == [
        "streamlit",
        "run",
        str(Path(module.__file__).parent / "app.py"),
    ]
    assert calls == ["run"]


def test_main_no_duplicate_src_path(monkeypatch) -> None:
    module = _reload_module()
    expected_src = str(Path(module.__file__).resolve().parents[2] / "src")

    calls: list[str] = []
    cli_module = ModuleType("streamlit.web.cli")
    cli_module.main = lambda: calls.append("run")
    web_module = ModuleType("streamlit.web")
    web_module.cli = cli_module
    streamlit_module = ModuleType("streamlit")
    streamlit_module.web = web_module
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setitem(sys.modules, "streamlit.web", web_module)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", cli_module)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    monkeypatch.setattr(module.sys, "argv", ["python"])
    monkeypatch.setattr(module.sys, "path", [expected_src, "existing"])

    module.main()

    assert module.sys.path[0] == expected_src
    assert module.sys.path.count(expected_src) == 1
    assert calls == ["run"]
