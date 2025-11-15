"""Results visualisation page for the Streamlit app."""

from __future__ import annotations

import io

import pandas as pd  # Required for type annotations used as strings
import streamlit as st

from trend_analysis.metrics import summary
from trend_analysis.viz import charts


def _create_line_chart_with_download(
    data_df: "pd.DataFrame", chart_title: str, button_label: str, filename: str
) -> None:
    """Create a line chart with CSV download functionality.

    Args:
        data_df: DataFrame containing the chart data
        chart_title: Title to display above the chart
        button_label: Label for the download button
        filename: Name of the CSV file for download
    """
    st.subheader(chart_title)
    st.line_chart(data_df)
    buf = io.StringIO()
    data_df.to_csv(buf)
    st.download_button(
        button_label,
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv",
    )


def _create_csv_download_button(
    data_df: "pd.DataFrame", button_label: str, filename: str
) -> None:
    """Create a CSV download button for data.

    Args:
        data_df: DataFrame containing the data
        button_label: Label for the download button
        filename: Name of the CSV file for download
    """
    buf = io.StringIO()
    data_df.to_csv(buf)
    st.download_button(
        button_label,
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv",
    )


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
_create_csv_download_button(sum_df, "Summary (CSV)", "summary.csv")
