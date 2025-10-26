from __future__ import annotations

import contextlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Callable

import pandas as pd
import pytest

import yaml
from trend_analysis.gui import app
from trend_analysis.gui.store import ParamStore


class DummyGrid:
    def __init__(self, df: pd.DataFrame, editable: bool = True) -> None:  # noqa: ARG002
        self.df = df
        self.data: list[dict[str, object]] = []
        self.layout = SimpleNamespace(border="")
        self._callbacks: dict[str, Callable[[dict[str, object]], None]] = {}
        created_instances["grid"] = self
        _instance_list("grids").append(self)

    def on(self, event: str, callback: Callable[[dict[str, object]], None]) -> None:
        self._callbacks[event] = callback

    def hold_trait_notifications(self):  # noqa: D401
        return contextlib.nullcontext()

    def trigger(self, event: str, payload: dict[str, object]) -> None:
        callback = self._callbacks.get(event)
        if callback is None:
            raise AssertionError(f"No callback registered for {event}")
        callback(payload)


class DummyUpload:
    def __init__(self, accept: str = "", multiple: bool = False) -> None:  # noqa: ARG002
        self.accept = accept
        self.multiple = multiple
        self.value: dict[str, dict[str, bytes]] = {}
        self._callbacks: list[Callable[[dict[str, object]], None]] = []
        created_instances["upload"] = self

    def observe(
        self, callback: Callable[[dict[str, object]], None], names: object | None = None
    ) -> None:
        self._callbacks.append(callback)

    def trigger(self, content: bytes) -> None:
        self.value = {"file": {"content": content}}
        for cb in self._callbacks:
            cb({"new": True})


class DummyDropdown:
    def __init__(self, options, value=None, description: str = "") -> None:  # noqa: ANN001, ARG002
        self.options = list(options)
        self.description = description
        default_value = self.options[0] if self.options else None
        self.value = value if value is not None else default_value
        self._callbacks: list[Callable[[dict[str, object]], None]] = []
        dropdowns = _instance_list("dropdowns")
        key = description or f"dropdown_{len(dropdowns)}"
        dropdowns.append(self)
        created_instances[key] = self

    def observe(
        self, callback: Callable[[dict[str, object]], None], names: object | None = None
    ) -> None:
        self._callbacks.append(callback)


class DummyCheckbox:
    def __init__(
        self, value: object = False, description: str = "", indent: bool = False
    ) -> None:  # noqa: ARG002
        self.value = value
        self.description = description
        self.indent = indent
        self._callbacks: list[Callable[[dict[str, object]], None]] = []
        _instance_list("checkboxes").append(self)

    def observe(
        self, callback: Callable[[dict[str, object]], None], names: object | None = None
    ) -> None:
        self._callbacks.append(callback)


class DummyToggleButtons(DummyCheckbox):
    def __init__(self, options, value=None, description: str = "") -> None:  # noqa: ANN001, ARG002
        super().__init__(
            value=value if value is not None else (options[0] if options else None),
            description=description,
        )
        self.options = options


class DummyButton:
    def __init__(self, description: str = "") -> None:  # noqa: ARG002
        self.description = description
        self._callbacks: list[Callable[..., None]] = []
        buttons = _instance_list("buttons")
        buttons.append(self)
        key = description or f"button_{len(buttons)}"
        created_instances[key] = self

    def on_click(self, callback: Callable[..., None]) -> None:
        self._callbacks.append(callback)


class DummyVBox:
    def __init__(self, children) -> None:  # noqa: ANN001
        self.children = children
        self.layout = SimpleNamespace(display="")


created_instances: dict[str, object] = {}


def _instance_list(key: str) -> list[object]:
    bucket = created_instances.get(key)
    if isinstance(bucket, list):
        return bucket
    new_list: list[object] = []
    created_instances[key] = new_list
    return new_list


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


def test_manual_override_grid_updates_state(monkeypatch):
    created_instances.clear()
    store = ParamStore(
        cfg={"portfolio": {"custom_weights": {"FundA": 0.25}, "manual_list": ["FundA"]}}
    )

    module = ModuleType("ipydatagrid")
    module.DataGrid = DummyGrid
    monkeypatch.setitem(sys.modules, "ipydatagrid", module)
    monkeypatch.setattr(app.widgets, "VBox", DummyVBox)

    widget = app._build_manual_override(store)

    assert isinstance(widget, DummyVBox)
    grid = widget.children[0]
    assert isinstance(grid, DummyGrid)

    callback = grid._callbacks.get("cell_edited")
    assert callback is not None

    store.dirty = False
    callback({"row": 0, "column": 1, "new": False})
    assert "FundA" not in store.cfg["portfolio"]["manual_list"]
    assert store.dirty is True

    store.dirty = False
    callback({"row": 0, "column": 1, "new": True})
    assert "FundA" in store.cfg["portfolio"]["manual_list"]
    assert store.dirty is True

    store.dirty = False
    callback({"row": 0, "column": 2, "new": None})
    assert store.cfg["portfolio"]["custom_weights"]["FundA"] == 0.25
    assert store.dirty is False

    store.dirty = False
    callback({"row": 0, "column": 2, "new": "invalid"})
    assert store.cfg["portfolio"]["custom_weights"]["FundA"] == 0.25
    assert store.dirty is False

    store.dirty = False
    callback({"row": 0, "column": 2, "new": "1.75"})
    assert store.cfg["portfolio"]["custom_weights"]["FundA"] == pytest.approx(1.75)
    assert grid.df.loc[0, "Weight"] == pytest.approx(1.75)
    assert store.dirty is True

    store.dirty = False
    callback({"row": 0, "column": 2, "new": "-0.5"})
    assert store.cfg["portfolio"]["custom_weights"]["FundA"] == pytest.approx(1.75)
    assert store.dirty is False
    assert widget.layout.display == "none"


class _FakePath:
    def __init__(self, name: str, behavior: object):
        self._name = name
        self._behavior = behavior

    def read_text(self) -> str:
        if isinstance(self._behavior, Exception):
            raise self._behavior
        return str(self._behavior)

    def __str__(self) -> str:
        return f"/fake/{self._name}"


class _FakeDir(dict[str, _FakePath]):
    def __truediv__(self, other: str) -> _FakePath:
        stem = other[:-4] if other.endswith(".yml") else other
        if stem not in self:
            raise FileNotFoundError(f"No such file: {other}")
        return self[stem]


def test_template_loader_handles_error_paths(monkeypatch, tmp_path):
    created_instances.clear()
    store = ParamStore(cfg={"foo": "bar"})

    fake_dir = _FakeDir(
        {
            "valid": _FakePath("valid.yml", "version: '1'\nalpha: 1"),
            "missing": _FakePath("missing.yml", FileNotFoundError("missing")),
            "permission": _FakePath("permission.yml", PermissionError("denied")),
            "invalid": _FakePath("invalid.yml", yaml.YAMLError("bad yaml")),
            "boom": _FakePath("boom.yml", RuntimeError("boom")),
        }
    )

    monkeypatch.setattr(app, "STATE_FILE", tmp_path / "state.yml")
    monkeypatch.setattr(app, "WEIGHT_STATE_FILE", tmp_path / "weights.pkl")
    monkeypatch.setattr(app, "DataGrid", DummyGrid)
    monkeypatch.setattr(app, "HAS_DATAGRID", True)
    monkeypatch.setattr(app.widgets, "FileUpload", DummyUpload)
    monkeypatch.setattr(app.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app.widgets, "Button", DummyButton)
    monkeypatch.setattr(app.widgets, "VBox", DummyVBox)
    monkeypatch.setattr(app.widgets, "HBox", DummyVBox)
    monkeypatch.setattr(app, "list_builtin_cfgs", lambda: list(fake_dir.keys()))
    monkeypatch.setattr(app, "_find_config_directory", lambda: fake_dir)

    reset_calls: list[ParamStore] = []
    monkeypatch.setattr(
        app, "reset_weight_state", lambda store: reset_calls.append(store)
    )

    widget = app._build_step0(store)
    assert isinstance(widget, DummyVBox)

    template = created_instances["Template"]
    grid = created_instances["grid"]
    callback = template._callbacks[0]

    callback({"new": "valid"})
    assert store.cfg == {"version": "1", "alpha": 1}
    assert grid.data == [store.cfg]
    assert reset_calls == [store]

    with pytest.warns(UserWarning, match="Template config file not found"):
        callback({"new": "missing"})

    with pytest.warns(UserWarning, match="Permission denied"):
        callback({"new": "permission"})

    with pytest.warns(UserWarning, match="Invalid YAML"):
        callback({"new": "invalid"})

    with pytest.warns(UserWarning, match="Failed to load template"):
        callback({"new": "boom"})


def test_launch_run_uses_registered_exporter(monkeypatch, tmp_path):
    created_instances.clear()
    store = ParamStore(
        cfg={
            "version": "1.0",
            "mode": "all",
            "output": {"format": "csv", "path": str(tmp_path / "out")},
        }
    )
    store.theme = "system"

    monkeypatch.setattr(app, "load_state", lambda: store)
    monkeypatch.setattr(app, "discover_plugins", lambda: None)
    monkeypatch.setattr(app.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app.widgets, "Checkbox", DummyCheckbox)
    monkeypatch.setattr(app.widgets, "ToggleButtons", DummyToggleButtons)
    monkeypatch.setattr(app.widgets, "Button", DummyButton)
    monkeypatch.setattr(app.widgets, "VBox", DummyVBox)
    monkeypatch.setattr(app.widgets, "HBox", DummyVBox)

    calls: list[tuple[dict[str, pd.DataFrame], str]] = []

    def fake_export(data, path, formatter):  # noqa: ANN001
        calls.append((data, path))

    monkeypatch.setitem(app.export.EXPORTERS, "csv", fake_export)
    monkeypatch.setattr(app, "save_state", lambda s: calls.append(({}, "saved")))
    monkeypatch.setattr(
        app.pipeline, "run", lambda cfg: pd.DataFrame({"metric": [1.0]})
    )
    monkeypatch.setattr(app.pipeline, "run_full", lambda cfg: {"extra": 1})

    widget = app.launch()
    assert isinstance(widget, DummyVBox)

    run_btn = created_instances.get("Run")
    assert run_btn is not None and run_btn._callbacks

    for cb in run_btn._callbacks:
        cb(run_btn)

    export_calls = [entry for entry in calls if "metrics" in entry[0]]
    assert export_calls, "Expected CSV exporter to be invoked"
    data, path = export_calls[-1]
    assert data["metrics"].equals(pd.DataFrame({"metric": [1.0]}))
    assert path == str(tmp_path / "out")
    assert store.dirty is False


def test_build_step0_handles_grid_without_data(monkeypatch, tmp_path):
    created_instances.clear()
    store = ParamStore(cfg={"foo": 1})

    class GridNoData:
        def __init__(self, df, editable=True):  # noqa: ANN001, ARG002
            self.df = df
            self.layout = SimpleNamespace(border="")
            created_instances["grid"] = self

        def on(self, *_, **__):  # noqa: ANN001, D401
            raise AttributeError("no event hooks")

        def hold_trait_notifications(self):  # noqa: D401
            return contextlib.nullcontext()

    monkeypatch.setattr(app, "STATE_FILE", tmp_path / "state.yml")
    monkeypatch.setattr(app, "WEIGHT_STATE_FILE", tmp_path / "weights.pkl")
    monkeypatch.setattr(app, "DataGrid", GridNoData)
    monkeypatch.setattr(app, "HAS_DATAGRID", True)
    monkeypatch.setattr(app.widgets, "FileUpload", DummyUpload)
    monkeypatch.setattr(app.widgets, "Dropdown", DummyDropdown)
    monkeypatch.setattr(app.widgets, "Button", DummyButton)
    monkeypatch.setattr(app.widgets, "VBox", DummyVBox)
    monkeypatch.setattr(app.widgets, "HBox", DummyVBox)
    monkeypatch.setattr(app, "list_builtin_cfgs", lambda: ["base"])

    widget = app._build_step0(store)
    assert isinstance(widget, DummyVBox)

    upload = created_instances["upload"]
    # Simulate a no-op update where change["new"] evaluates to False
    callback = upload._callbacks[0]
    store.dirty = False
    callback({"new": False})
    assert store.cfg == {"foo": 1}
    assert store.dirty is False

    payload = yaml.safe_dump({"alpha": 2}).encode("utf-8")
    upload.trigger(payload)

    assert store.cfg == {"alpha": 2}
    assert store.dirty is True
    grid = created_instances["grid"]
    assert not hasattr(grid, "data"), "Guard should skip grids without data attribute"
