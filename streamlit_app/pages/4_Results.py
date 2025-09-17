import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from trend_analysis.engine.walkforward import walk_forward
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


st.title("Results")

if "sim_results" not in st.session_state:
    st.error("Run a simulation first.")
    st.stop()

res = st.session_state["sim_results"]

if st.session_state.pop("demo_show_export_prompt", False):
    st.success("Demo run complete! Download bundle from the Export tab when ready.")
    if st.button("Go to Export", key="btn_demo_go_export", type="primary"):
        st.switch_page("pages/5_Export.py")
        st.stop()

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
    st.line_chart(res.portfolio_curve())
with c2:
    st.subheader("Drawdown")
    st.line_chart(res.drawdown_curve())

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
        st.area_chart(w_df)
    else:
        st.caption("No weights recorded.")
except (KeyError, ValueError, TypeError, AttributeError, ImportError):
    st.caption("Weights view unavailable.")

st.subheader("Event log")
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
                    st.bar_chart(chart_df)
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
                        st.line_chart(chart_df)
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
                        st.bar_chart(chart_df)
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
            st.line_chart(cum)
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
