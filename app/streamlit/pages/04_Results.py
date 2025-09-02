"""Results visualisation page for the Streamlit app."""

from __future__ import annotations

import io

import streamlit as st

from trend_analysis.viz import charts
from trend_analysis.metrics import summary


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
    ec_df = charts.equity_curve(returns)
    _create_line_chart_with_download(
        ec_df, "Equity curve", "Equity curve (CSV)", "equity_curve.csv"
    )
with c2:
    dd_df = charts.drawdown_curve(returns)
    _create_line_chart_with_download(
        dd_df, "Drawdown", "Drawdown (CSV)", "drawdown_curve.csv"
    )

c3, c4 = st.columns(2)
with c3:
    ir_df = charts.rolling_information_ratio(returns, benchmark)
    _create_line_chart_with_download(
        ir_df, "Rolling info ratio", "Rolling IR (CSV)", "rolling_ir.csv"
    )
with c4:
    to_df = charts.turnover_series(weights)
    _create_line_chart_with_download(
        to_df, "Turnover", "Turnover (CSV)", "turnover.csv"
    )

st.subheader("Portfolio weights")
w_df = charts.weights_heatmap_data(weights)
st.dataframe(w_df.style.background_gradient(cmap="viridis"))
_create_csv_download_button(w_df, "Weights (CSV)", "weights.csv")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader("Summary")
sum_df = summary.summary_table(returns, weights, benchmark)
st.table(sum_df)
_create_csv_download_button(sum_df, "Summary (CSV)", "summary.csv")
