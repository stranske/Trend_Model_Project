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
from .plugins import discover_plugins, iter_plugins
from .utils import list_builtin_cfgs, debounce
from ..config import Config
from .. import pipeline, export, weighting

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


def _build_rank_options(store: ParamStore) -> widgets.Widget:
    """Return widgets for ranking configuration (Step 2)."""
    from ..core.rank_selection import METRIC_REGISTRY

    rank_cfg = store.cfg.setdefault("rank", {})

    incl_dd = widgets.Dropdown(
        options=["top_n", "top_pct", "threshold"],
        value=rank_cfg.get("inclusion_approach", "top_n"),
        description="Approach",
    )
    metric_dd = widgets.Dropdown(
        options=list(METRIC_REGISTRY) + ["blended"],
        value=rank_cfg.get("score_by", "Sharpe"),
        description="Score By",
    )
    n_int = widgets.BoundedIntText(value=rank_cfg.get("n", 8), min=1, description="N")
    pct_flt = widgets.BoundedFloatText(
        value=rank_cfg.get("pct", 0.10),
        min=0.01,
        max=1.0,
        step=0.01,
        description="Pct",
    )
    thresh_f = widgets.FloatText(
        value=rank_cfg.get("threshold", 1.0), description="Threshold"
    )

    try:
        weights = rank_cfg.get("blended_weights", {})
        items = list(METRIC_REGISTRY)
        m1_dd = widgets.Dropdown(options=items, value=items[0], description="M1")
        w1_sl = widgets.FloatSlider(
            value=weights.get(items[0], 0.33), min=0, max=1, step=0.01
        )
        m2_dd = widgets.Dropdown(
            options=items,
            value=items[1] if len(items) > 1 else items[0],
            description="M2",
        )
        w2_sl = widgets.FloatSlider(
            value=weights.get(items[1] if len(items) > 1 else items[0], 0.33),
            min=0,
            max=1,
            step=0.01,
        )
        m3_dd = widgets.Dropdown(
            options=items,
            value=items[2] if len(items) > 2 else items[0],
            description="M3",
        )
        w3_sl = widgets.FloatSlider(
            value=weights.get(items[2] if len(items) > 2 else items[0], 0.34),
            min=0,
            max=1,
            step=0.01,
        )
        blended_box = widgets.VBox([m1_dd, w1_sl, m2_dd, w2_sl, m3_dd, w3_sl])
    except Exception:  # pragma: no cover - never raised
        blended_box = widgets.VBox()

    def _store_rank(_: Any = None) -> None:
        rank_cfg["inclusion_approach"] = incl_dd.value
        rank_cfg["score_by"] = metric_dd.value
        rank_cfg["n"] = int(n_int.value)
        rank_cfg["pct"] = float(pct_flt.value)
        rank_cfg["threshold"] = float(thresh_f.value)
        rank_cfg["blended_weights"] = {
            m1_dd.value: w1_sl.value,
            m2_dd.value: w2_sl.value,
            m3_dd.value: w3_sl.value,
        }
        store.dirty = True

    incl_dd.observe(_store_rank, names="value")
    metric_dd.observe(_store_rank, names="value")
    n_int.observe(_store_rank, names="value")
    pct_flt.observe(_store_rank, names="value")
    thresh_f.observe(_store_rank, names="value")

    @debounce(300)
    def _on_blend(_: Any) -> None:
        _store_rank()

    m1_dd.observe(_on_blend, names="value")
    w1_sl.observe(_on_blend, names="value")
    m2_dd.observe(_on_blend, names="value")
    w2_sl.observe(_on_blend, names="value")
    m3_dd.observe(_on_blend, names="value")
    w3_sl.observe(_on_blend, names="value")

    metric_dd.observe(
        lambda change: blended_box.layout.__setattr__(
            "display", "flex" if change["new"] == "blended" else "none"
        ),
        names="value",
    )
    blended_box.layout.display = "none" if metric_dd.value != "blended" else "flex"

    return widgets.VBox([incl_dd, metric_dd, n_int, pct_flt, thresh_f, blended_box])


def _build_manual_override(store: ParamStore) -> widgets.Widget:
    """Return manual-selection grid or fallback."""
    port = store.cfg.setdefault("portfolio", {})
    weights = port.setdefault("custom_weights", {})
    manual = port.setdefault("manual_list", list(weights))

    try:
        from ipydatagrid import DataGrid

        rows = [
            {"Fund": f, "Include": f in manual, "Weight": float(weights.get(f, 0))}
            for f in sorted(set(manual) | set(weights))
        ]
        df = pd.DataFrame(rows, columns=["Fund", "Include", "Weight"])
        grid = DataGrid(df, editable=True)

        def _on_edit(event: dict[str, Any]) -> None:
            fund = df.loc[event["row"], "Fund"]
            if event.get("column") == 1:  # Include
                include_val = bool(event.get("new"))
                if include_val and fund not in manual:
                    manual.append(fund)
                elif not include_val and fund in manual:
                    manual.remove(fund)
            elif event.get("column") == 2:  # Weight
                new_val = event.get("new")
                if new_val is None:
                    return
                try:
                    weight_val = float(new_val)
                    if weight_val < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    return
                weights[fund] = weight_val
                df.loc[event["row"], "Weight"] = weight_val
            store.dirty = True

        grid.on("cell_edited", _on_edit)
        box = widgets.VBox([grid])
    except Exception:  # pragma: no cover - optional dep
        opts = sorted(set(manual) | set(weights))
        warn = widgets.Label("ipydatagrid not installed")
        select = widgets.SelectMultiple(options=opts, value=tuple(manual))
        weight_boxes = [
            widgets.FloatText(value=float(weights.get(f, 0)), description=f)
            for f in opts
        ]

        def _on_select(change: dict[str, Any]) -> None:
            manual[:] = list(change["new"])
            store.dirty = True

        def _on_weight(change: dict[str, Any], fund: str) -> None:
            try:
                val = float(change["new"])
                if val < 0:
                    raise ValueError
            except Exception:
                return
            weights[fund] = val
            store.dirty = True

        select.observe(_on_select, names="value")
        for wdg in weight_boxes:
            wdg.observe(lambda ch, fund=wdg.description: _on_weight(ch, fund), names="value")

        box = widgets.VBox([warn, select] + weight_boxes)

    box.layout.display = "none"
    return box


def _build_weighting_options(store: ParamStore) -> widgets.Widget:
    """Return weighting method dropdown and param sliders."""
    weight_cfg = store.cfg.setdefault("portfolio", {}).setdefault(
        "weighting", {"name": "equal", "params": {}}
    )
    options = {
        "equal": weighting.EqualWeight,
        "score_prop": weighting.ScorePropSimple,
        "score_prop_bayes": weighting.ScorePropBayesian,
        "adaptive_bayes": weighting.AdaptiveBayesWeighting,
    }
    for plugin in iter_plugins():
        name = plugin.__name__.lower()
        options[name] = plugin

    method_dd = widgets.Dropdown(
        options=list(options),
        value=weight_cfg.get("name", "equal"),
        description="Weighting",
    )
    params = weight_cfg.setdefault("params", {})
    hl = widgets.IntSlider(
        value=params.get("half_life", 90), min=30, max=365, description="half_life"
    )
    os_sl = widgets.FloatSlider(
        value=params.get("obs_sigma", 0.25),
        min=0.0,
        max=1.0,
        step=0.01,
        description="obs_sigma",
    )
    mw_sl = widgets.FloatSlider(
        value=params.get("max_w", 0.20),
        min=0.0,
        max=0.5,
        step=0.01,
        description="max_w",
    )
    pt_sl = widgets.FloatSlider(
        value=params.get("prior_tau", 1.0),
        min=0.0,
        max=5.0,
        step=0.1,
        description="prior_tau",
    )
    adv_box = widgets.VBox([hl, os_sl, mw_sl, pt_sl])

    def _store_weight(_: Any = None) -> None:
        weight_cfg["name"] = method_dd.value
        params["half_life"] = int(hl.value)
        params["obs_sigma"] = float(os_sl.value)
        params["max_w"] = float(mw_sl.value)
        params["prior_tau"] = float(pt_sl.value)
        store.dirty = True

    method_dd.observe(_store_weight, names="value")

    @debounce(300)
    def _on_param(_: Any) -> None:
        _store_weight()

    for wdg in (hl, os_sl, mw_sl, pt_sl):
        wdg.observe(_on_param, names="value")

    def _toggle_adv(change: dict[str, Any]) -> None:
        adv_box.layout.display = "flex" if change["new"] == "adaptive_bayes" else "none"

    method_dd.observe(_toggle_adv, names="value")
    _toggle_adv({"new": method_dd.value})

    return widgets.VBox([method_dd, adv_box])


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

    rank_box = _build_rank_options(store)
    manual_box = _build_manual_override(store)
    weight_box = _build_weighting_options(store)

    def _toggle_boxes(change: dict[str, Any]) -> None:
        mode_val = change["new"] if isinstance(change, dict) else mode.value
        rank_box.layout.display = (
            "flex" if mode_val == "rank" or use_ranking.value else "none"
        )
        manual_box.layout.display = "flex" if mode_val == "manual" else "none"

    mode.observe(_toggle_boxes, names="value")
    use_ranking.observe(_toggle_boxes, names="value")
    _toggle_boxes({"new": mode.value})

    step0 = _build_step0(store)

    container = widgets.VBox(
        [
            step0,
            mode,
            vol_adj,
            use_ranking,
            rank_box,
            manual_box,
            weight_box,
            fmt_dd,
            theme,
            run_btn,
        ]
    )
    return container


__all__ = [
    "launch",
    "build_config_dict",
    "build_config_from_store",
    "load_state",
    "save_state",
]
