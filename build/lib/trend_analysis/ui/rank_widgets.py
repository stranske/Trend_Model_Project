from __future__ import annotations

import io
from typing import Any, Callable, cast

import ipywidgets as widgets
import pandas as pd

from .. import export, pipeline
from ..core.rank_selection import METRIC_REGISTRY
from ..data import ensure_datetime, load_csv
from ..export import Formatter

# ===============================================================
#  UI SCAFFOLD (very condensed – Codex expands)
# ===============================================================


def build_ui() -> widgets.VBox:  # pragma: no cover - UI wiring exercised manually
    # -------------------- Step 1: data source & periods --------------------
    source_tb = widgets.ToggleButtons(
        options=["Path/URL", "Browse"],
        description="Source:",
    )
    csv_path = widgets.Text(description="CSV or URL:")
    file_up = widgets.FileUpload(accept=".csv", multiple=False)
    file_up.layout.display = "none"
    load_btn = widgets.Button(description="Load CSV", button_style="success")
    load_out = widgets.Output()

    in_start = widgets.Text(description="In Start:")
    in_end = widgets.Text(description="In End:")
    out_start = widgets.Text(description="Out Start:")
    out_end = widgets.Text(description="Out End:")

    session: dict[str, Any] = {"df": None, "rf": None}
    idx_select = widgets.SelectMultiple(options=[], description="Indices:")
    idx_select.layout.display = "none"
    bench_select = widgets.SelectMultiple(options=[], description="Benchmarks:")
    bench_select.layout.display = "none"
    step1_box = widgets.VBox(
        [
            source_tb,
            csv_path,
            file_up,
            load_btn,
            load_out,
            idx_select,
            bench_select,
            in_start,
            in_end,
            out_start,
            out_end,
        ]
    )

    def _load_action(_btn: widgets.Button) -> None:
        with load_out:
            load_out.clear_output()
            try:
                df: pd.DataFrame | None = None
                if source_tb.value == "Browse":
                    if not file_up.value:
                        print("Upload a CSV")
                        return
                    # ipywidgets 7.x returns a dict; 8.x returns a tuple
                    if isinstance(file_up.value, dict):
                        item = next(iter(file_up.value.values()))
                    else:
                        item = file_up.value[0]
                    df = pd.read_csv(io.BytesIO(item["content"]))
                else:
                    path = csv_path.value.strip()
                    if not path:
                        print("Enter CSV path or URL")
                        return
                    if path.startswith("http://") or path.startswith("https://"):
                        df = pd.read_csv(path)
                    else:
                        df = load_csv(path)
                if df is None:
                    print("Failed to load")
                    return
                df = ensure_datetime(df)
                session["df"] = df
                # session["rf"] = rf  # rf is not defined here, skip or set to None
                dates = df["Date"].dt.to_period("M")
                in_start.value = str(dates.min())
                in_end.value = str(dates.min() + 2)
                out_start.value = str(dates.min() + 3)
                out_end.value = str(dates.min() + 5)
                idx_select.options = [c for c in df.columns if c not in {"Date"}]
                idx_select.layout.display = "flex"
                bench_select.options = [c for c in df.columns if c not in {"Date"}]
                bench_select.layout.display = "flex"
                print(f"Loaded {len(df):,} rows")
            except Exception as exc:
                session["df"] = None
                print("Error:", exc)

    load_btn.on_click(_load_action)

    def _source_toggle(*_: Any) -> None:
        if source_tb.value == "Browse":
            file_up.layout.display = "flex"
            csv_path.layout.display = "none"
        else:
            file_up.layout.display = "none"
            csv_path.layout.display = "flex"

    source_tb.observe(_source_toggle, "value")
    _source_toggle()

    # -------------------- Step 2: selection & ranking ----------------------
    mode_dd = widgets.Dropdown(options=["all", "random", "manual", "rank"], description="Mode:")
    random_n_int = widgets.BoundedIntText(value=8, min=1, description="Random N:")
    random_n_int.layout.display = "none"
    vol_ck = widgets.Checkbox(value=True, description="Vol‑adjust?")
    target_vol = widgets.BoundedFloatText(
        value=1.0, min=0.05, max=3.0, step=0.01, description="Target Vol:"
    )
    use_rank_ck = widgets.Checkbox(value=True, description="Apply ranking?")

    incl_dd = widgets.Dropdown(
        options=["top_n", "top_pct", "threshold"], value="top_n", description="Approach"
    )
    metric_dd = widgets.Dropdown(
        options=list(METRIC_REGISTRY) + ["blended"], value="Sharpe", description="Score"
    )
    topn_int = widgets.BoundedIntText(value=10, min=1, description="N:")
    pct_flt = widgets.BoundedFloatText(value=0.1, min=0.01, max=1.0, step=0.01, description="Pct:")
    thresh_f = widgets.FloatText(value=1.0, description="Threshold:")
    m1_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M1")
    w1_sl = widgets.FloatSlider(value=0.33, min=0, max=1.0, step=0.01)
    m2_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M2")
    w2_sl = widgets.FloatSlider(value=0.33, min=0, max=1.0, step=0.01)
    m3_dd = widgets.Dropdown(options=list(METRIC_REGISTRY), description="M3")
    w3_sl = widgets.FloatSlider(value=0.34, min=0, max=1.0, step=0.01)

    out_fmt = widgets.Dropdown(
        options=["excel", "csv", "json"], value="excel", description="Output:"
    )

    blended_box = widgets.VBox([m1_dd, w1_sl, m2_dd, w2_sl, m3_dd, w3_sl])
    rank_box = widgets.VBox(
        [
            widgets.HTML("<h4>Ranking</h4>"),
            widgets.HBox([incl_dd, metric_dd]),
            widgets.HBox([topn_int, pct_flt, thresh_f]),
            blended_box,
        ]
    )

    def _update_rank_vis(*_: Any) -> None:
        blended_box.layout.display = "flex" if metric_dd.value == "blended" else "none"
        incl = incl_dd.value
        topn_int.layout.display = "flex" if incl == "top_n" else "none"
        pct_flt.layout.display = "flex" if incl == "top_pct" else "none"
        thresh_f.layout.display = "flex" if incl == "threshold" else "none"

    def _update_random_vis(*_: Any) -> None:
        random_n_int.layout.display = "flex" if mode_dd.value == "random" else "none"

    def _update_target_vol(*_: Any) -> None:
        target_vol.layout.display = "flex" if vol_ck.value else "none"

    def _toggle_rank_fields(*_: Any) -> None:
        rank_box.layout.display = "flex" if use_rank_ck.value else "none"

    def _on_mode_change(change: dict[str, Any]) -> None:
        _update_random_vis()
        _toggle_rank_fields()
        use_rank_ck.value = use_rank_ck.value or change["new"] == "rank"

    incl_dd.observe(_update_rank_vis, names="value")
    metric_dd.observe(_update_rank_vis, names="value")
    mode_dd.observe(_on_mode_change, names="value")
    use_rank_ck.observe(_toggle_rank_fields, names="value")
    vol_ck.observe(_update_target_vol, names="value")

    _update_rank_vis()
    _update_random_vis()
    _toggle_rank_fields()
    _update_target_vol()

    # -------------------- Step 3: manual override --------------------
    manual_box = widgets.VBox()
    manual_scores_html = widgets.HTML()
    manual_checks: list[widgets.Checkbox] = []
    manual_weights: list[widgets.FloatText] = []
    manual_total_lbl = widgets.Label("Total weight: 0 %")

    def _update_manual() -> None:
        df = session.get("df")
        if df is None:
            manual_box.children = [widgets.Label("Load data first")]
            return
        res = pipeline.run_analysis(
            df,
            in_start.value,
            in_end.value,
            out_start.value,
            out_end.value,
            target_vol.value if vol_ck.value else 1.0,
            0.0,
            selection_mode="all",
            indices_list=list(idx_select.value),
            benchmarks={b: b for b in bench_select.value},
        )
        diag = res.diagnostic if res else None
        if res and res.value:
            scores = res.value["scores"]
            manual_checks.clear()
            manual_weights.clear()
            rows: list[widgets.Widget] = []
            manual_funds: list[str] = []
            for f, score in scores.items():
                chk = widgets.Checkbox(value=False, description=f)
                wt = widgets.FloatText(value=0.0, layout=widgets.Layout(width="80px"))
                manual_checks.append(chk)
                manual_weights.append(wt)
                manual_funds.append(f)
                rows.append(widgets.HBox([chk, wt]))
            manual_box.children = [widgets.HTML("<h4>Manual override</h4>")] + rows
            manual_scores_html.value = scores.to_frame("Score").to_html()
        else:
            reason = "No scores" if diag is None else diag.message
            manual_box.children = [widgets.Label(f"No scores ({reason})")]
            manual_scores_html.value = ""
        manual_total_lbl.value = "Total weight: 0 %"

    def _update_inclusion_fields(change: dict[str, Any] | None = None) -> None:
        if change is not None and change.get("name") != "value":
            return
        _update_rank_vis()

    incl_dd.observe(_update_inclusion_fields, names="value")
    metric_dd.observe(_update_inclusion_fields, names="value")

    def _on_mode(change: dict[str, Any]) -> None:
        _update_random_vis()
        _update_rank_vis()
        manual_scores_html.layout.display = "flex" if change["new"] == "manual" else "none"
        manual_box.layout.display = "flex" if change["new"] == "manual" else "none"

    mode_dd.observe(_on_mode, names="value")
    _on_mode({"new": mode_dd.value})
    use_rank_ck.observe(lambda ch: _update_rank_vis(), names="value")
    vol_ck.observe(lambda ch: _update_target_vol(), names="value")

    # -------------------- Step 4: run button + output --------------------
    custom_weights: dict[str, float] = {}
    manual_funds: list[str] = []

    rank_kwargs: dict[str, Callable[[], Any]] = {
        "use_ranking": lambda: use_rank_ck.value or mode_dd.value == "rank",
        "inclusion_approach": lambda: incl_dd.value,
        "n": lambda: int(topn_int.value),
        "pct": lambda: float(pct_flt.value),
        "threshold": lambda: float(thresh_f.value),
        "score_by": lambda: metric_dd.value,
        "blended_weights": lambda: {
            m1_dd.value: w1_sl.value,
            m2_dd.value: w2_sl.value,
            m3_dd.value: w3_sl.value,
        },
    }

    run_btn = widgets.Button(description="Run")
    next_btn_1 = widgets.Button(description="Next")
    output = widgets.Output()

    def _collect_manual_weights() -> None:
        manual_funds.clear()
        custom_weights.clear()
        for chk, wt in zip(manual_checks, manual_weights):
            if chk.value:
                manual_funds.append(chk.description)
                custom_weights[chk.description] = float(wt.value)
        total = sum(custom_weights.values())
        manual_total_lbl.value = f"Total weight: {total:.2%}"

    def _on_manual_change(_change: dict[str, Any]) -> None:
        _collect_manual_weights()

    def _on_next(_: widgets.Button) -> None:
        _collect_manual_weights()
        _update_manual()

    for chk in manual_checks:
        chk.observe(_on_manual_change, names="value")
    for wt in manual_weights:
        wt.observe(_on_manual_change, names="value")

    next_btn_1.on_click(_on_next)

    def _run_action(_btn: widgets.Button) -> None:
        manual_funds.clear()
        custom_weights.clear()
        for chk, wt in zip(manual_checks, manual_weights):
            if chk.value:
                manual_funds.append(chk.description)
                custom_weights[chk.description] = float(wt.value)

        with output:
            output.clear_output()
            try:
                df = session.get("df")
                if df is None:
                    print("Load data first")
                    return

                mode = mode_dd.value
                if mode_dd.value == "manual" and not custom_weights:
                    print("No funds selected")
                    return

                res = pipeline.run_analysis(
                    df,
                    in_start.value,
                    in_end.value,
                    out_start.value,
                    out_end.value,
                    target_vol.value if vol_ck.value else 1.0,
                    0.0,
                    selection_mode=mode,
                    random_n=int(random_n_int.value),
                    custom_weights=custom_weights,
                    rank_kwargs={k: v() for k, v in rank_kwargs.items()},
                    manual_funds=manual_funds,
                    indices_list=list(idx_select.value),
                    benchmarks={b: b for b in bench_select.value},
                )
                if not res:
                    diag = res.diagnostic
                    if diag:
                        print(f"No results ({diag.reason_code}: {diag.message})")
                    else:
                        print("No results")
                else:
                    payload = res.value or {}
                    sheet_formatter = export.make_summary_formatter(
                        payload,
                        in_start.value,
                        in_end.value,
                        out_start.value,
                        out_end.value,
                    )
                    text = export.format_summary_text(
                        payload,
                        in_start.value,
                        in_end.value,
                        out_start.value,
                        out_end.value,
                    )
                    print(text)
                    data = {"summary": pd.DataFrame()}
                    prefix = f"IS_{in_start.value}_OS_{out_start.value}"
                    export.export_data(
                        data,
                        prefix,
                        formats=[out_fmt.value],
                        formatter=cast(Formatter, sheet_formatter),
                    )
            except Exception as exc:
                print("Error:", exc)

    run_btn.on_click(_run_action)

    ui = widgets.VBox(
        [
            step1_box,
            mode_dd,
            random_n_int,
            vol_ck,
            target_vol,
            use_rank_ck,
            next_btn_1,
            rank_box,
            manual_box,
            out_fmt,
            run_btn,
            output,
        ]
    )
    _update_rank_vis()
    _update_inclusion_fields()

    _update_random_vis()

    _update_manual()
    _update_target_vol()

    return ui
