import streamlit as st

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

st.subheader("Event log")
st.dataframe(res.event_log_df().tail(200))

st.subheader("Summary")
st.json(res.summary())
