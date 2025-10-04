from __future__ import annotations
import io
import json
import hashlib
import os
from pathlib import Path as _Path

import altair as alt
import numpy as np
import pandas as pd
from pandas.api import types as ptypes
import streamlit as st

from trend.reporting import generate_unified_report
from trend_analysis.engine.walkforward import walk_forward
from trend_analysis.io import export_bundle
from trend_analysis.logging import error_summary, logfile_to_frame
from trend_analysis.metrics import attribution


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        flat_cols = []
        for col in out.columns:
            parts = [str(part) for part in col if part not in (None, "")]
            if parts and parts[0] == "window":
                parts = parts[1:]
            flat_cols.append(" • ".join(parts) if parts else "")
        out.columns = flat_cols
        return out
    return df


CHART_PALETTE = ["#2563EB", "#EA580C", "#16A34A", "#9333EA", "#0891B2", "#F59E0B"]


def _prepare_chart_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str, bool]:
    frame = df.copy()
    index_name = frame.index.name or "Date"
    frame = frame.reset_index()
    is_datetime = ptypes.is_datetime64_any_dtype(frame[index_name])
    if not is_datetime:
        try:
            frame[index_name] = pd.to_datetime(frame[index_name])
        except Exception:
            pass
        else:
            is_datetime = True
    return frame.rename(columns={index_name: "Date"}), "Date", is_datetime


def _render_single_line(
    series: pd.Series | pd.DataFrame,
    *,
    label: str,
    color: str = CHART_PALETTE[0],
    y_title: str,
) -> None:
    frame = series.to_frame(name=label) if isinstance(series, pd.Series) else series
    frame = frame.rename(columns={frame.columns[0]: label})
    prepared, date_col, is_datetime = _prepare_chart_frame(frame)
    chart = (
        alt.Chart(prepared)
        .mark_line(color=color, strokeWidth=2)
        .encode(
            x=alt.X(
                f"{date_col}:{'T' if is_datetime else 'O'}",
                title="Date" if is_datetime else date_col,
            ),
            y=alt.Y(f"{label}:Q", title=y_title),
            tooltip=[f"{date_col}:{'T' if is_datetime else 'O'}", alt.Tooltip(f"{label}:Q", title=y_title)],
        )
    )
    st.altair_chart(chart.interactive(), use_container_width=True)


def _render_multi_line(df: pd.DataFrame, *, y_title: str) -> None:
    if df.empty:
        st.caption("No data available.")
        return
    prepared, date_col, is_datetime = _prepare_chart_frame(df)
    melted = prepared.melt(id_vars=[date_col], var_name="Series", value_name="value")
    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                f"{date_col}:{'T' if is_datetime else 'O'}",
                title="Date" if is_datetime else date_col,
            ),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("Series:N", scale=alt.Scale(range=CHART_PALETTE)),
            tooltip=[
                f"{date_col}:{'T' if is_datetime else 'O'}",
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("value:Q", title=y_title),
            ],
        )
    )
    st.altair_chart(chart.interactive(), use_container_width=True)


def _render_stacked_area(df: pd.DataFrame, *, y_title: str) -> None:
    if df.empty:
        st.caption("No data available.")
        return
    prepared, date_col, is_datetime = _prepare_chart_frame(df)
    melted = prepared.melt(id_vars=[date_col], var_name="Series", value_name="value")
    chart = (
        alt.Chart(melted)
        .mark_area(opacity=0.75)
        .encode(
            x=alt.X(
                f"{date_col}:{'T' if is_datetime else 'O'}",
                title="Date" if is_datetime else date_col,
            ),
            y=alt.Y("value:Q", title=y_title, stack="normalize" if "Weight" in y_title else "zero"),
            color=alt.Color("Series:N", scale=alt.Scale(range=CHART_PALETTE)),
            tooltip=[
                f"{date_col}:{'T' if is_datetime else 'O'}",
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("value:Q", title=y_title),
            ],
        )
    )
    st.altair_chart(chart.interactive(), use_container_width=True)


def _render_bar(df: pd.DataFrame, *, value_label: str) -> None:
    if df.empty:
        st.caption("No data available.")
        return
    frame = df.reset_index()
    if frame.shape[1] == 2:
        category_col = frame.columns[0]
        value_col = frame.columns[1]
        renamed = frame.rename(columns={category_col: "Category", value_col: "value"})
        chart = (
            alt.Chart(renamed)
            .mark_bar(color=CHART_PALETTE[2])
            .encode(
                x=alt.X("Category:N", title=""),
                y=alt.Y("value:Q", title=value_label),
                tooltip=[
                    alt.Tooltip("Category:N", title="Category"),
                    alt.Tooltip("value:Q", title=value_label),
                ],
            )
        )
    else:
        category = frame.columns[0]
        melted = frame.melt(id_vars=[category], var_name="Series", value_name="value")
        chart = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                x=alt.X("Series:N", title="Series"),
                y=alt.Y("value:Q", title=value_label),
                color=alt.Color("Series:N", scale=alt.Scale(range=CHART_PALETTE)),
                column=alt.Column(f"{category}:N", title=category),
                tooltip=[
                    alt.Tooltip(f"{category}:N", title=category),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("value:Q", title=value_label),
                ],
            )
        )
    st.altair_chart(chart.interactive(), use_container_width=True)


def _generate_cache_key(results, config_dict) -> str:
    portfolio_hash = hashlib.sha256(results.portfolio.to_csv().encode("utf-8")).hexdigest()[
        :16
    ]
    config_hash = hashlib.sha256(str(sorted(config_dict.items())).encode("utf-8")).hexdigest()[
        :16
    ]
    return f"export_bundle_{portfolio_hash}_{config_hash}"


@st.cache_data(ttl=300)
def _cached_export_bundle(
    results_cache_key: str, config_dict, _results
) -> tuple[bytes, str]:
    path = export_bundle(_results, config_dict)
    with open(path, "rb") as handle:
        bundle_data = handle.read()
    return bundle_data, os.path.basename(path)
st.title("Results")

if "sim_results" not in st.session_state:
    st.error("Run a simulation first.")
    st.stop()

res = st.session_state["sim_results"]

if st.session_state.pop("demo_show_export_prompt", False):
    st.success("Demo run complete! Scroll to Downloads to grab the bundle and reports.")

# One-time banner for weight engine fallback
try:
    fb_info = getattr(res, "fallback_info", None)
except Exception:  # pragma: no cover - defensive
    fb_info = None
if fb_info and not st.session_state.get("dismiss_weight_engine_fallback"):
    with st.warning(
        "⚠️ Weight engine '%s' failed (%s). Portfolio uses equal weights."
        % (fb_info.get("engine"), fb_info.get("error_type"))
    ):
        if st.button(
            "Dismiss",
            key="btn_dismiss_weight_engine_fallback_results",
            help="Hide this warning banner.",
        ):
            st.session_state["dismiss_weight_engine_fallback"] = True
            try:
                st.rerun()
            except Exception:  # pragma: no cover
                pass

c1, c2 = st.columns(2)
with c1:
    st.subheader("Equity curve")
    show_band = st.toggle(
        "Show bootstrap uncertainty band",
        value=False,
        help="Estimate a 5–95% confidence band via block bootstrap sampling.",
    )
    curve = res.portfolio_curve()
    if curve.empty:
        st.caption("No equity data available.")
    else:
        base_frame = curve.to_frame("Equity")
        prepared, date_col, is_datetime = _prepare_chart_frame(base_frame)
        x_encoding = alt.X(
            f"{date_col}:{'T' if is_datetime else 'O'}",
            title="Date" if is_datetime else date_col,
        )
        base_chart = (
            alt.Chart(prepared)
            .mark_line(color=CHART_PALETTE[0], strokeWidth=2)
            .encode(
                x=x_encoding,
                y=alt.Y("Equity:Q", title="Equity"),
                tooltip=[
                    f"{date_col}:{'T' if is_datetime else 'O'}",
                    alt.Tooltip("Equity:Q", title="Equity"),
                ],
            )
        )
        chart = base_chart
        bootstrap_error: str | None = None
        if show_band:
            try:
                band = res.bootstrap_band()
                if band is None or band.empty:
                    raise ValueError("bootstrap returned no data")
                band = band.reindex(curve.index)
                valid = band[["p05", "p95"]].notna().all(axis=1)
                if not valid.any():
                    raise ValueError("insufficient in-sample history")
                band_valid = band.loc[valid, ["p05", "p95", "median"]]
                band_frame, band_date_col, band_is_datetime = _prepare_chart_frame(
                    band_valid
                )
                band_x = alt.X(
                    f"{band_date_col}:{'T' if band_is_datetime else 'O'}",
                    title="Date" if band_is_datetime else band_date_col,
                )
                band_chart = (
                    alt.Chart(band_frame)
                    .mark_area(color=CHART_PALETTE[0], opacity=0.18)
                    .encode(
                        x=band_x,
                        y=alt.Y("p05:Q", title="Equity"),
                        y2="p95:Q",
                    )
                )
                median_chart = (
                    alt.Chart(band_frame)
                    .mark_line(color=CHART_PALETTE[0], strokeDash=[6, 4])
                    .encode(
                        x=band_x,
                        y=alt.Y("median:Q", title="Equity"),
                        tooltip=[
                            f"{band_date_col}:{'T' if band_is_datetime else 'O'}",
                            alt.Tooltip("median:Q", title="Bootstrap median"),
                        ],
                    )
                )
                chart = band_chart + median_chart + base_chart
            except Exception as exc:  # pragma: no cover - defensive UX branch
                bootstrap_error = str(exc)
        st.altair_chart(chart.interactive(), use_container_width=True)
        if bootstrap_error:
            st.info(f"Bootstrap band unavailable: {bootstrap_error}")
with c2:
    st.subheader("Drawdown")
    drawdown = res.drawdown_curve()
    if drawdown.empty:
        st.caption("No drawdown data available.")
    else:
        _render_single_line(drawdown, label="Drawdown", color=CHART_PALETTE[1], y_title="Drawdown")

st.subheader("Weights")
try:
    # Build weights history into a table: index = dates, columns = managers
    w_df = None
    if hasattr(res, "weights") and isinstance(res.weights, dict) and res.weights:
        w_df = (
            (  # type: ignore[assignment]
                pd.DataFrame({d: s for d, s in res.weights.items()})
            )
            .T.sort_index()
            .fillna(0.0)
        )
    if w_df is not None and not w_df.empty:
        _render_stacked_area(w_df, y_title="Portfolio Weight")
    else:
        st.caption("No weights recorded.")
except (KeyError, ValueError, TypeError, AttributeError, ImportError):
    st.caption("Weights view unavailable.")

st.subheader("Run Log")

run_id = getattr(res, "seed", None)  # fallback if run_id not attached
log_path_str = None
try:
    log_path_str = st.session_state.get("run_log_path")
except Exception:  # pragma: no cover
    log_path_str = None
if log_path_str:
    lp = _Path(log_path_str)
    # Auto-refresh every 5 seconds (limit 600 refreshes ~ 50 min session)
    st.autorefresh(interval=5000, limit=600, key="runlog_autorefresh")
    df_log = logfile_to_frame(lp, limit=500)
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        size_kb = lp.stat().st_size / 1024 if lp.exists() else 0
        st.caption(f"Log file: {lp} ({size_kb:.1f} KB)")
    with c2:
        if st.button("Manual refresh", key="btn_refresh_log"):
            st.experimental_rerun()
    with c3:
        st.caption(f"Lines: {len(df_log)}")
    if not df_log.empty:
        st.dataframe(
            df_log[[c for c in ["ts", "step", "level", "msg"] if c in df_log.columns]]
        )
        errs = error_summary(lp)
        if not errs.empty:
            st.markdown("**Errors**")
            st.dataframe(errs)
        # Download button
        try:
            st.download_button(
                label="Download JSONL",
                data=lp.read_bytes(),
                file_name=lp.name,
                mime="application/json",
            )
        except Exception:  # pragma: no cover
            pass
    else:
        st.caption("No structured log lines yet.")
else:
    st.caption("No structured log available for this run.")

st.subheader("Event log (legacy in-memory)")
st.dataframe(res.event_log_df().tail(200))

st.subheader("Summary")
summary = res.summary()
st.json(summary)

# Walk-forward and regime analysis
st.subheader("Walk-forward analysis")

with st.expander("Run walk-forward (rolling OOS) analysis"):
    # Inputs for window sizes and regimes
    c1, c2, c3 = st.columns(3)
    with c1:
        train_size = st.number_input("Train size (rows)", min_value=1, value=12, step=1)
    with c2:
        test_size = st.number_input("Test size (rows)", min_value=1, value=3, step=1)
    with c3:
        step_size = st.number_input("Step (rows)", min_value=1, value=3, step=1)

    # Optional regime source: either none or infer simple regimes by sign of benchmark/portfolio
    regime_source = st.selectbox(
        "Regime labels",
        (
            "None",
            "Portfolio sign (+/-)",
            "Upload CSV",
        ),
        index=0,
    )

    # Build a simple DataFrame with a metric to aggregate. Use portfolio returns if available.
    try:
        portfolio_curve = res.portfolio_curve()
        wf_df = pd.DataFrame(
            {"Date": portfolio_curve.index, "metric": portfolio_curve.values}
        )

        # Only proceed with walk-forward analysis if we have valid data
        if not wf_df.empty and len(wf_df.columns) >= 2:
            regimes = None
            regime_error: str | None = None
            if regime_source == "Portfolio sign (+/-)":
                try:
                    s = wf_df.set_index("Date").iloc[:, 0]
                    regimes = pd.Series(np.where(s >= 0, "+", "-"), index=s.index)
                except Exception as exc:
                    regime_error = str(exc)
                    regimes = None
            elif regime_source == "Upload CSV":
                regime_file = st.file_uploader(
                    "Regime CSV (requires a Date column)",
                    type=["csv"],
                    key="wf_regime_csv",
                )
                if regime_file is not None:
                    try:
                        reg_df = pd.read_csv(regime_file)
                        date_col = None
                        if "Date" in reg_df.columns:
                            date_col = "Date"
                        else:
                            for c in reg_df.columns:
                                if c.lower().startswith("date"):
                                    date_col = c
                                    break
                        if date_col is None:
                            raise ValueError("CSV must contain a Date column")
                        reg_df[date_col] = pd.to_datetime(reg_df[date_col])
                        label_cols = [c for c in reg_df.columns if c != date_col]
                        if not label_cols:
                            raise ValueError("No label column found in CSV")
                        regime_col = st.selectbox(
                            "Regime column",
                            options=label_cols,
                            key="wf_regime_column",
                        )
                        regimes = reg_df.set_index(date_col)[regime_col]
                    except (
                        pd.errors.EmptyDataError,
                        ValueError,
                        FileNotFoundError,
                    ) as exc:  # pragma: no cover - streamlit UI feedback
                        regime_error = str(exc)
                        regimes = None
                else:
                    st.caption("Upload a CSV with columns 'Date' and regime labels.")
            if regime_error:
                st.warning(f"Regime data unavailable: {regime_error}")

            metric_name = wf_df.columns[1]
            res_wf = walk_forward(
                wf_df,
                train_size=train_size,
                test_size=test_size,
                step_size=step_size,
                metric_cols=[metric_name],
                regimes=regimes,
                agg="mean",
            )

            view = st.radio(
                "View",
                ("Full period", "OOS only", "Per regime"),
                horizontal=True,
            )

            st.caption(f"Detected periods/year: {res_wf.periods_per_year}")

            if view == "Full period":
                st.write("Full-period summary:")
                st.dataframe(res_wf.full)
                stat_options = list(res_wf.full.index)
                if stat_options:
                    full_stat = st.selectbox(
                        "Statistic",
                        stat_options,
                        key="wf_full_stat",
                    )
                    chart_df = res_wf.full.loc[[full_stat]].T
                    chart_df.columns = [full_stat]
                    _render_bar(chart_df, value_label=full_stat)
            elif view == "OOS only":
                st.write("Out-of-sample summary:")
                st.dataframe(res_wf.oos)
                if not res_wf.oos_windows.empty:
                    st.write("Per-window breakdown:")
                    st.dataframe(_flatten_columns(res_wf.oos_windows))

                    metric_cols = res_wf.oos_windows.loc[
                        :,
                        res_wf.oos_windows.columns.get_level_values("category")
                        != "window",
                    ]
                    stat_options = list(
                        dict.fromkeys(metric_cols.columns.get_level_values("statistic"))
                    )
                    if stat_options:
                        oos_stat = st.selectbox(
                            "Statistic",
                            stat_options,
                            key="wf_oos_stat",
                        )
                        chart_df = metric_cols.xs(oos_stat, axis=1, level="statistic")
                        if ("window", "test_end") in res_wf.oos_windows.columns:
                            chart_df = chart_df.copy()
                            chart_df.index = res_wf.oos_windows["window", "test_end"]
                            chart_df.index.name = "test_end"
                        _render_multi_line(chart_df, y_title=oos_stat)
                else:
                    st.caption("No OOS windows generated for the selected parameters.")
            else:
                st.write("Per-regime aggregate (OOS windows):")
                if res_wf.by_regime is not None and not res_wf.by_regime.empty:
                    st.dataframe(_flatten_columns(res_wf.by_regime))
                    stat_options = list(
                        dict.fromkeys(
                            res_wf.by_regime.columns.get_level_values("statistic")
                        )
                    )
                    if stat_options:
                        regime_stat = st.selectbox(
                            "Statistic",
                            stat_options,
                            key="wf_regime_stat",
                        )
                        chart_df = res_wf.by_regime.xs(
                            regime_stat, axis=1, level="statistic"
                        )
                        _render_bar(chart_df, value_label=regime_stat)
                else:
                    st.caption("No regime data available.")
        else:
            st.caption("No data available for walk-forward analysis.")

    except (AttributeError, KeyError, ValueError, TypeError) as e:
        st.warning(f"Walk-forward data unavailable: {e}")
        st.caption("No data available for walk-forward analysis.")

# ---------------------------------------------------------------------------
# Performance attribution (optional upload or auto-detect)
# ---------------------------------------------------------------------------
st.subheader("Performance attribution")
with st.expander("Compute contributions by signal and rebalancing"):
    st.caption(
        "Provide per-signal PnL as a CSV (columns = signals, index/date in first column) "
        "and a rebalancing PnL CSV (single column). If available in the result object, "
        "we'll use those automatically."
    )

    # Try to auto-detect on the result object first
    auto_signals = None
    auto_rebal = None
    try:
        if hasattr(res, "signal_pnls") and isinstance(res.signal_pnls, pd.DataFrame):
            auto_signals = res.signal_pnls.copy()
        if hasattr(res, "rebalancing_pnl") and isinstance(
            res.rebalancing_pnl, pd.Series
        ):
            auto_rebal = res.rebalancing_pnl.copy()
    except Exception:
        auto_signals = None
        auto_rebal = None

    # Upload fallbacks if auto not present
    sig_file = st.file_uploader("Signal PnL CSV", type=["csv"], key="attr_signals")
    reb_file = st.file_uploader("Rebalancing PnL CSV", type=["csv"], key="attr_rebal")

    signals_df = None
    rebal_s = None
    try:
        if auto_signals is not None:
            signals_df = auto_signals
        elif sig_file is not None:
            signals_df = pd.read_csv(sig_file, index_col=0, parse_dates=True)

        if auto_rebal is not None:
            rebal_s = auto_rebal
        elif reb_file is not None:
            rb = pd.read_csv(reb_file, index_col=0, parse_dates=True)
            # Use first column as the series
            if isinstance(rb, pd.DataFrame) and rb.shape[1] >= 1:
                first_col = rb.columns[0]
                rebal_s = rb[first_col]
            elif isinstance(rb, pd.Series):
                rebal_s = rb
    except pd.errors.EmptyDataError as e:
        st.warning(f"Uploaded file is empty: {e}")
        signals_df = None
        rebal_s = None
    except ValueError as e:
        st.warning(f"Failed to parse uploaded files (ValueError): {e}")
        signals_df = None
        rebal_s = None
    except TypeError as e:
        st.warning(f"Failed to parse uploaded files (TypeError): {e}")
        signals_df = None
        rebal_s = None
    except FileNotFoundError as e:
        st.warning(f"File not found: {e}")
        signals_df = None
        rebal_s = None
    except Exception as e:  # pragma: no cover - UI path
        st.warning(f"Unexpected error while parsing uploaded files: {e}")
        signals_df = None
        rebal_s = None

    if signals_df is None or rebal_s is None:
        st.info(
            "Waiting for both Signal PnL and Rebalancing PnL inputs (or auto-detected values)."
        )
    else:
        try:
            contrib = attribution.compute_contributions(signals_df, rebal_s)
            st.success("Contributions computed.")

            # Chart (cumulative) and table
            cum = contrib.drop(
                columns=[c for c in ["total"] if c in contrib.columns]
            ).cumsum()
            _render_multi_line(cum, y_title="Cumulative contribution")
            st.dataframe(contrib.tail(20))

            # Download CSV
            buf = io.StringIO()
            contrib.to_csv(buf)
            st.download_button(
                "Contributions (CSV)",
                data=buf.getvalue(),
                file_name="contributions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.warning(f"Attribution failed: {e}")

st.subheader("Downloads")
bundle_data = None
bundle_filename = None
bundle_error: str | None = None
cfg = st.session_state.get("sim_config")
if cfg is not None:
    try:
        cache_key = _generate_cache_key(res, cfg)
        bundle_data, bundle_filename = _cached_export_bundle(cache_key, cfg, res)
    except Exception as exc:  # pragma: no cover - defensive guard
        bundle_error = str(exc)
else:
    bundle_error = "Configuration payload missing; rerun the model to enable bundle download."

if bundle_data and bundle_filename:
    bundle_size = len(bundle_data) / (1024 * 1024)
    st.download_button(
        label=f"Results bundle (ZIP) — {bundle_size:.2f} MB",
        data=bundle_data,
        file_name=bundle_filename,
        mime="application/zip",
        type="primary",
    )
elif bundle_error:
    st.info(f"Bundle unavailable: {bundle_error}")

col1, col2, col3 = st.columns(3)
with col1:
    csv_buf = io.StringIO()
    res.portfolio.to_csv(csv_buf, header=["return"])  # type: ignore[attr-defined]
    st.download_button(
        label="Portfolio returns (CSV)",
        data=csv_buf.getvalue(),
        file_name="portfolio_returns.csv",
        mime="text/csv",
    )
with col2:
    ev_csv = io.StringIO()
    res.event_log_df().to_csv(ev_csv)
    st.download_button(
        label="Event log (CSV)",
        data=ev_csv.getvalue(),
        file_name="event_log.csv",
        mime="text/csv",
    )
with col3:
    st.download_button(
        label="Summary (JSON)",
        data=json.dumps(summary, indent=2),
        file_name="summary.json",
        mime="application/json",
    )

report_error: str | None = None
report_artifacts = None
try:
    cfg_payload = st.session_state.get("sim_config")
except Exception:  # pragma: no cover - defensive access
    cfg_payload = None

run_identifier = getattr(res, "run_id", None)
if not run_identifier:
    run_identifier = getattr(res, "seed", "app")
run_identifier = str(run_identifier)

try:
    report_artifacts = generate_unified_report(
        res,
        cfg_payload if cfg_payload is not None else {},
        run_id=run_identifier,
        include_pdf=True,
    )
except RuntimeError as exc:
    report_error = str(exc)
    try:
        report_artifacts = generate_unified_report(
            res,
            cfg_payload if cfg_payload is not None else {},
            run_id=run_identifier,
            include_pdf=False,
        )
    except Exception as inner_exc:  # pragma: no cover - defensive fallback
        report_artifacts = None
        report_error = str(inner_exc)
except Exception as exc:  # pragma: no cover - defensive fallback
    report_artifacts = None
    report_error = str(exc)

if report_artifacts is not None:
    st.download_button(
        label="Download report (HTML)",
        data=report_artifacts.html.encode("utf-8"),
        file_name=f"trend_report_{run_identifier}.html",
        mime="text/html",
    )
    if report_artifacts.pdf_bytes:
        st.download_button(
            label="Download report (PDF)",
            data=report_artifacts.pdf_bytes,
            file_name=f"trend_report_{run_identifier}.pdf",
            mime="application/pdf",
        )
elif report_error:
    st.info(f"Report download unavailable: {report_error}")
