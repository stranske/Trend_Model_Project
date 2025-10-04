# --- Streamlit UI ---
import streamlit as st

from streamlit_app.components.demo_runner import run_one_click_demo

st.set_page_config(
    page_title="Trend Portfolio Simulator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.title("Trend Portfolio Simulator")
st.markdown(
    """
Upload a CSV of manager returns and step through **Data → Model → Results**.
The new layout guides you from dataset validation to configuration and analysis without touching Python.
    """
)
st.info("Start with the Data tab to upload or load the bundled sample dataset.")

st.markdown("---")
st.subheader("Quick start")
if st.button("🚀 Run demo", type="primary"):
    with st.spinner("Loading demo data and running the analysis..."):
        success = run_one_click_demo()
    if success:
        st.success("Demo ready! Redirecting to Results…")
        st.switch_page("pages/3_Results.py")
