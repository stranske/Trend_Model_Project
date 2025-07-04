from __future__ import annotations

from pathlib import Path
import warnings
import yaml
import ipywidgets as widgets
from typing import Any

from .store import ParamStore
from .plugins import discover_plugins
from .utils import list_builtin_cfgs

STATE_FILE = Path.home() / ".trend_gui_state.yml"


def load_state() -> ParamStore:
    """Load persisted GUI state from ``STATE_FILE`` if possible."""
    try:
        if STATE_FILE.exists():
            return ParamStore.from_yaml(STATE_FILE)
    except Exception as exc:  # pragma: no cover - malformed file
        warnings.warn(f"Failed to load state: {exc}")
    return ParamStore()


def save_state(store: ParamStore) -> None:
    """Persist ``store`` to ``STATE_FILE``."""
    STATE_FILE.write_text(yaml.safe_dump(store.to_dict()))


def build_config_dict(store: ParamStore) -> dict[str, object]:
    """Return the config dictionary kept in ``store``."""
    return dict(store.cfg)


def _build_step0(store: ParamStore) -> widgets.Widget:
    """Return widgets for Step 0 (config loader/editor)."""

    upload = widgets.FileUpload(accept=".yml", multiple=False)
    template = widgets.Dropdown(options=list_builtin_cfgs(), description="Template")
    try:
        from ipydatagrid import DataGrid  # type: ignore

        grid = DataGrid([], disabled=False)
    except Exception:  # pragma: no cover - optional dep
        grid = widgets.Label("ipydatagrid not installed")

    save_btn = widgets.Button(description="💾 Save config")
    download_btn = widgets.Button(description="⬇️ Download")

    def refresh_grid() -> None:
        if hasattr(grid, "data"):
            with grid.hold_trait_notifications():  # type: ignore[attr-defined]
                grid.data = [store.cfg]

    def on_upload(change: dict[str, Any]) -> None:
        if change["new"]:
            item = next(iter(upload.value.values()))
            store.cfg = yaml.safe_load(item["content"].decode("utf-8"))
            store.dirty = True
            refresh_grid()

    def on_template(change: dict[str, Any]) -> None:
        name = change["new"]
        cfg_dir = Path(__file__).resolve().parents[2] / "config"
        path = cfg_dir / f"{name}.yml"
        store.cfg = yaml.safe_load(path.read_text())
        store.dirty = True
        refresh_grid()

    def on_save(_: Any) -> None:
        save_state(store)
        store.dirty = False

    upload.observe(on_upload, names="value")
    template.observe(on_template, names="value")
    save_btn.on_click(on_save)

    return widgets.VBox(
        [template, upload, grid, widgets.HBox([save_btn, download_btn])]
    )


def launch() -> widgets.Widget:
    """Return the root widget for the Trend Model GUI."""
    store = load_state()
    discover_plugins()

    mode = widgets.Dropdown(
        options=["all", "random", "manual", "rank"],
        value=store.cfg.get("mode", "all"),
        description="Mode",
    )
    theme = widgets.ToggleButtons(
        options=["system", "light", "dark"],
        value=store.theme,
        description="Theme",
    )

    def on_theme(change: dict[str, Any]) -> None:
        store.theme = change["new"]
        store.dirty = True

    theme.observe(on_theme, names="value")

    def on_mode(change: dict[str, Any]) -> None:
        store.cfg["mode"] = change["new"]
        store.dirty = True

    mode.observe(on_mode, names="value")

    step0 = _build_step0(store)

    container = widgets.VBox([step0, mode, theme])
    return container


__all__ = ["launch", "build_config_dict", "load_state", "save_state"]
