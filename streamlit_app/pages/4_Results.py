import streamlit as st
import io
import json
import numpy as np
import pandas as pd
from trend_analysis.engine.walkforward import walk_forward

st.title("Results")

if "sim_results" not in st.session_state:
    st.error("Run a simulation first.")
    st.stop()

res = st.session_state["sim_results"]

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
        ),
        index=0,
    )

    # Build a simple DataFrame with a metric to aggregate. Use portfolio returns if available.
    try:
        portfolio_curve = res.portfolio_curve()
        wf_df = pd.DataFrame({
            "Date": portfolio_curve.index,
            "metric": portfolio_curve.values
        })
        
        # Only proceed with walk-forward analysis if we have valid data
        if not wf_df.empty and len(wf_df.columns) >= 2:
            regimes = None
            if regime_source == "Portfolio sign (+/-)":
                try:
                    s = wf_df.set_index("Date").iloc[:, 0]
                    regimes = pd.Series(np.where(s >= 0, "+", "-"), index=s.index)
                except Exception:
                    regimes = None

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

            if view == "Full period":
                st.write("Full-period aggregate:")
                st.dataframe(res_wf.full.to_frame("mean"))
            elif view == "OOS only":
                st.write("Out-of-sample aggregate:")
                st.dataframe(res_wf.oos.to_frame("mean"))
            else:
                st.write("Per-regime aggregate (OOS windows):")
                if res_wf.by_regime is not None and not res_wf.by_regime.empty:
                    st.dataframe(res_wf.by_regime)
                else:
                    st.caption("No regime data available.")
        else:
            st.caption("No data available for walk-forward analysis.")
            
    except (AttributeError, KeyError, ValueError, TypeError) as e:
        st.warning(f"Walk-forward data unavailable: {e}")
        st.caption("No data available for walk-forward analysis.")

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
