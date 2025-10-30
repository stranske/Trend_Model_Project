import asyncio
import importlib
import sys
from types import SimpleNamespace

import yaml

import trend_analysis.gui as gui


def test_paramstore_roundtrip(tmp_path):
    cfg = {"a": 1}
    path = tmp_path / "s.yml"
    path.write_text(yaml.safe_dump(cfg))
    store = gui.ParamStore.from_yaml(path)
    assert store.cfg == cfg
    assert store.to_dict() == cfg


def test_debounce():
    calls = []

    @gui.debounce(10)
    async def foo(x):
        calls.append(x)

    async def run():
        await asyncio.gather(foo(1), foo(2))
        await asyncio.sleep(0.02)

    asyncio.run(run())
    assert calls == [2]


def test_plugin_discovery(monkeypatch):
    dummy = type("Dummy", (), {})
    eps = [SimpleNamespace(load=lambda: dummy)]

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None: eps)
    gui.plugins._PLUGIN_REGISTRY.clear()

    gui.discover_plugins()

    assert dummy in list(gui.iter_plugins())


def test_list_builtin_cfgs():
    cfgs = gui.list_builtin_cfgs()
    assert "defaults" in cfgs


def test_state_persistence(tmp_path, monkeypatch):
    path = tmp_path / "state.yml"
    pkl = path.with_suffix(".pkl")
    monkeypatch.setattr(gui.app, "STATE_FILE", path)
    monkeypatch.setattr(gui.app, "WEIGHT_STATE_FILE", pkl)
    store = gui.ParamStore(cfg={"x": 1}, weight_state={"a": 1.0})
    gui.save_state(store)
    assert pkl.exists()
    loaded = gui.load_state()
    assert loaded.cfg == {"x": 1}
    assert loaded.weight_state == {"a": 1.0}


def test_build_config_from_store():
    cfg = {
        "version": "1",
        "data": {"csv_path": "foo.csv"},
        "preprocessing": {},
        "vol_adjust": {},
        "sample_split": {},
        "portfolio": {},
        "metrics": {},
        "export": {},
        "run": {},
    }
    store = gui.ParamStore(cfg=cfg)
    out = gui.build_config_from_store(store)
    assert out.data["csv_path"] == "foo.csv"


def test_manual_override_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "ipydatagrid", None)
    from ipywidgets import Label, VBox

    store = gui.ParamStore(cfg={})
    widget = gui.app._build_manual_override(store)
    assert isinstance(widget, VBox)
    assert isinstance(widget.children[0], Label)


def test_reset_weight_state(tmp_path, monkeypatch):
    path = tmp_path / "state.yml"
    pkl = path.with_suffix(".pkl")
    monkeypatch.setattr(gui.app, "STATE_FILE", path)
    monkeypatch.setattr(gui.app, "WEIGHT_STATE_FILE", pkl)
    store = gui.ParamStore(cfg={}, weight_state={"a": 1})
    gui.save_state(store)
    assert pkl.exists()
    gui.reset_weight_state(store)
    assert store.weight_state is None
    assert not pkl.exists()
