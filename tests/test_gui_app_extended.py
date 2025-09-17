from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pandas as pd
import yaml

from trend_analysis.gui import app
from trend_analysis.gui.store import ParamStore


class DummyGrid:
    def __init__(self, df: pd.DataFrame, editable: bool = True) -> None:  # noqa: ARG002
        self.df = df
        self.data: list[dict[str, object]] = []
        self.layout = SimpleNamespace(border="")
        self._callbacks: dict[str, object] = {}
        created_instances["grid"] = self

    def on(self, event: str, callback) -> None:  # noqa: ANN001
        self._callbacks[event] = callback

    def hold_trait_notifications(self):  # noqa: D401
        return contextlib.nullcontext()


class DummyUpload:
    def __init__(self, accept: str = "", multiple: bool = False) -> None:  # noqa: ARG002
        self.accept = accept
        self.multiple = multiple
        self.value: dict[str, dict[str, bytes]] = {}
        self._callbacks: list = []
        created_instances["upload"] = self

    def observe(self, callback, names=None) -> None:  # noqa: ANN001
        self._callbacks.append(callback)

    def trigger(self, content: bytes) -> None:
        self.value = {"file": {"content": content}}
        for cb in self._callbacks:
            cb({"new": True})


class DummyDropdown:
    def __init__(self, options, description: str = "") -> None:  # noqa: ANN001, ARG002
        self.options = options
        self.description = description
        self.value = options[0] if options else None
        self._callbacks: list = []

    def observe(self, callback, names=None) -> None:  # noqa: ANN001
        self._callbacks.append(callback)


class DummyButton:
    def __init__(self, description: str = "") -> None:  # noqa: ARG002
        self.description = description
        self._callbacks: list = []

    def on_click(self, callback) -> None:  # noqa: ANN001
        self._callbacks.append(callback)


class DummyVBox:
    def __init__(self, children) -> None:  # noqa: ANN001
        self.children = children


created_instances: dict[str, object] = {}


def test_build_step0_upload_refresh(monkeypatch, tmp_path):
    created_instances.clear()
    store = ParamStore(cfg={"foo": 1})

    monkeypatch.setattr(app, "STATE_FILE", tmp_path / "state.yml")
    monkeypatch.setattr(app, "WEIGHT_STATE_FILE", tmp_path / "weights.pkl")
    monkeypatch.setattr(app, "DataGrid", DummyGrid)
    monkeypatch.setattr(app, "HAS_DATAGRID", True)
    monkeypatch.setattr(app, "list_builtin_cfgs", lambda: ["base"])
    monkeypatch.setattr(app.widgets, "FileUpload", DummyUpload)
    monkeypatch.setattr(app.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app.widgets, "Button", DummyButton)
    monkeypatch.setattr(app.widgets, "VBox", DummyVBox)
    monkeypatch.setattr(app.widgets, "HBox", DummyVBox)

    widget = app._build_step0(store)
    assert isinstance(widget, DummyVBox)

    upload = created_instances["upload"]
    grid = created_instances["grid"]
    payload = yaml.safe_dump({"alpha": 2}).encode("utf-8")
    upload.trigger(payload)

    assert store.cfg == {"alpha": 2}
    assert store.dirty is True
    assert grid.data == [store.cfg]
