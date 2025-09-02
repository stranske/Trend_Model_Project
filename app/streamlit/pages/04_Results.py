"""Results visualisation page for the Streamlit app."""

from __future__ import annotations

import io

import streamlit as st

from trend_analysis.viz import charts
from trend_analysis.metrics import summary


st.title("Results")

if "sim_results" not in st.session_state:
    st.error("Run a simulation first.")
    st.stop()

res = st.session_state["sim_results"]
returns = res.portfolio
benchmark = getattr(res, "benchmark", None)
weights = getattr(res, "weights", {})

# ---------------------------------------------------------------------------
# Core charts
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Equity curve")
    fig, ec_df = charts.equity_curve(returns)
    st.pyplot(fig)
    buf = io.StringIO()
    ec_df.to_csv(buf)
    st.download_button(
        "Equity curve (CSV)",
        data=buf.getvalue(),
        file_name="equity_curve.csv",
        mime="text/csv",
    )
with c2:
    st.subheader("Drawdown")
    fig, dd_df = charts.drawdown_curve(returns)
    st.pyplot(fig)
    buf = io.StringIO()
    dd_df.to_csv(buf)
    st.download_button(
        "Drawdown (CSV)",
        data=buf.getvalue(),
        file_name="drawdown_curve.csv",
        mime="text/csv",
    )

c3, c4 = st.columns(2)
with c3:
    st.subheader("Rolling info ratio")
    fig, ir_df = charts.rolling_information_ratio(returns, benchmark)
    st.pyplot(fig)
    buf = io.StringIO()
    ir_df.to_csv(buf)
    st.download_button(
        "Rolling IR (CSV)",
        data=buf.getvalue(),
        file_name="rolling_ir.csv",
        mime="text/csv",
    )
with c4:
    st.subheader("Turnover")
    fig, to_df = charts.turnover_series(weights)
    st.pyplot(fig)
    buf = io.StringIO()
    to_df.to_csv(buf)
    st.download_button(
        "Turnover (CSV)",
        data=buf.getvalue(),
        file_name="turnover.csv",
        mime="text/csv",
    )

st.subheader("Portfolio weights")
fig, w_df = charts.weights_heatmap(weights)
st.pyplot(fig)
buf = io.StringIO()
w_df.to_csv(buf)
st.download_button(
    "Weights (CSV)",
    data=buf.getvalue(),
    file_name="weights.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader("Summary")
sum_df = summary.summary_table(returns, weights, benchmark)
st.table(sum_df)
buf = io.StringIO()
sum_df.to_csv(buf)
st.download_button(
    "Summary (CSV)",
    data=buf.getvalue(),
    file_name="summary.csv",
    mime="text/csv",
)
