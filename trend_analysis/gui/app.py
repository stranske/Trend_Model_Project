from __future__ import annotations

from pathlib import Path
import warnings
import yaml
import ipywidgets as widgets
from typing import Any

from .store import ParamStore
from .plugins import discover_plugins

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

    def on_mode(change: dict[str, Any]) -> None:
        store.cfg["mode"] = change["new"]
        store.dirty = True

    mode.observe(on_mode, names="value")

    container = widgets.VBox([mode, theme])
    return container


__all__ = ["launch", "build_config_dict", "load_state", "save_state"]
