import streamlit as st
import io
import json
import pandas as pd

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
