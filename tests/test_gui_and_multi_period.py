import importlib
import sys
import types

import pandas as pd

from trend_analysis.multi_period import engine, replacer


def test_engine_run_placeholder():
    cfg = {"foo": "bar"}
    assert engine.run(cfg) == {}


def test_rebalancer_apply_triggers_returns_copy():
    prev = pd.Series({"A": 0.6, "B": 0.4})
    scores = pd.DataFrame({"m": [1.0, 2.0]}, index=["A", "B"])
    r = replacer.Rebalancer({})
    out = r.apply_triggers(prev, scores)
    assert out.equals(prev)
    assert out is not prev


def test_gui_app_import(monkeypatch):
    calls = {}

    def title(msg: str) -> None:
        calls["title"] = msg

    def write(msg: str) -> None:
        calls["write"] = msg

    dummy = types.SimpleNamespace(title=title, write=write)
    monkeypatch.setitem(sys.modules, "streamlit", dummy)
    monkeypatch.delitem(sys.modules, "trend_analysis.gui.app", raising=False)
    importlib.import_module("trend_analysis.gui.app")
    assert "GUI coming soon" in calls["title"]
    assert "placeholder" in calls["write"]
