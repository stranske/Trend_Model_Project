from __future__ import annotations

import asyncio
import warnings
import yaml
import ipywidgets as widgets
from IPython.display import Javascript, display, FileLink
from typing import Any, cast
import pandas as pd

from pathlib import Path
from .store import ParamStore
from .plugins import discover_plugins
from .utils import list_builtin_cfgs
from ..config import Config
from .. import pipeline, export

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


def build_config_from_store(store: ParamStore) -> Config:
    """Convert ``store`` into a :class:`Config` object."""
    return Config(**build_config_dict(store))


def _build_step0(store: ParamStore) -> widgets.Widget:
    """Return widgets for Step 0 (config loader/editor)."""

    upload = widgets.FileUpload(accept=".yml", multiple=False)
    template = widgets.Dropdown(options=list_builtin_cfgs(), description="Template")
    try:
        from ipydatagrid import DataGrid

        grid_df = pd.DataFrame(list(store.cfg.items()), columns=["Key", "Value"])
        grid = DataGrid(grid_df, editable=True)

        def on_cell_change(event: dict[str, Any]) -> None:
            if event.get("column") != 1:  # value column only
                return
            key = grid_df.iloc[event["row"], 0]
            old = grid_df.iloc[event["row"], 1]
            new = event["new"]
            try:
                store.cfg[key] = yaml.safe_load(new)
                grid_df.iloc[event["row"], 1] = new
                store.dirty = True
            except Exception:
                grid_df.iloc[event["row"], 1] = old
                grid.layout.border = "2px solid red"
                asyncio.get_event_loop().call_later(
                    1.0, lambda: setattr(grid.layout, "border", "")
                )

        grid.on("cell_edited", on_cell_change)
    except Exception:  # pragma: no cover - optional dep
        grid = widgets.Label("ipydatagrid not installed")

    save_btn = widgets.Button(description="💾 Save config")
    download_btn = widgets.Button(description="⬇️ Download")

    def refresh_grid() -> None:
        if hasattr(grid, "data"):
            with grid.hold_trait_notifications():
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

    def on_download(_: Any) -> None:
        path = STATE_FILE.with_name("config_download.yml")
        path.write_text(yaml.safe_dump(store.to_dict()))
        cast(Any, display)(cast(Any, FileLink)(path))

    upload.observe(on_upload, names="value")
    template.observe(on_template, names="value")
    save_btn.on_click(on_save)
    download_btn.on_click(on_download)

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
    vol_adj = widgets.Checkbox(
        value=store.cfg.get("use_vol_adjust", False),
        description="Vol-Adj",
        indent=False,
    )
    use_ranking = widgets.Checkbox(
        value=store.cfg.get("use_ranking", False),
        description="Use Ranking",
        indent=False,
    )
    theme = widgets.ToggleButtons(
        options=["system", "light", "dark"],
        value=store.theme,
        description="Theme",
    )
    fmt_dd = widgets.Dropdown(
        options=["excel", "csv", "json"],
        value=store.cfg.get("output", {}).get("format", "excel"),
        description="Format",
    )
    run_btn = widgets.Button(description="Run")

    def on_theme(change: dict[str, Any]) -> None:
        store.theme = change["new"]
        store.dirty = True
        js = cast(Any, Javascript)(
            f"document.documentElement.style.setProperty('--trend-theme', '{change['new']}');"
        )
        cast(Any, display)(js)

    theme.observe(on_theme, names="value")

    def on_mode(change: dict[str, Any]) -> None:
        store.cfg["mode"] = change["new"]
        store.dirty = True

    mode.observe(on_mode, names="value")

    def on_vol(change: dict[str, Any]) -> None:
        store.cfg["use_vol_adjust"] = bool(change["new"])
        store.dirty = True

    def on_rank(change: dict[str, Any]) -> None:
        store.cfg["use_ranking"] = bool(change["new"])
        store.dirty = True

    def on_fmt(change: dict[str, Any]) -> None:
        out = store.cfg.setdefault("output", {})
        out["format"] = change["new"]
        store.dirty = True

    def on_run(_: Any) -> None:
        cfg = build_config_from_store(store)
        metrics = pipeline.run(cfg)
        if metrics.empty:
            return
        out = cfg.output or {}
        fmt = out.get("format", "excel").lower()
        path = out.get("path", "gui_output")
        data = {"metrics": metrics}
        if fmt in {"excel", "xlsx"}:
            res = pipeline.run_full(cfg)
            split = cfg.sample_split
            sheet_fmt = export.make_summary_formatter(
                res,
                str(split.get("in_start", "")),
                str(split.get("in_end", "")),
                str(split.get("out_start", "")),
                str(split.get("out_end", "")),
            )
            data["summary"] = pd.DataFrame()
            export.export_to_excel(
                data,
                str(Path(path).with_suffix(".xlsx")),
                default_sheet_formatter=sheet_fmt,
            )
        elif fmt in export.EXPORTERS:
            export.EXPORTERS[fmt](data, path, None)
        save_state(store)
        store.dirty = False

    vol_adj.observe(on_vol, names="value")
    use_ranking.observe(on_rank, names="value")
    fmt_dd.observe(on_fmt, names="value")
    run_btn.on_click(on_run)

    step0 = _build_step0(store)

    container = widgets.VBox(
        [step0, mode, vol_adj, use_ranking, fmt_dd, theme, run_btn]
    )
    return container


__all__ = [
    "launch",
    "build_config_dict",
    "build_config_from_store",
    "load_state",
    "save_state",
]
