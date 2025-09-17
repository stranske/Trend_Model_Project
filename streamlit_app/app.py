# --- Streamlit UI ---
import streamlit as st

from streamlit_app.demo_runner import run_one_click_demo

st.set_page_config(
    page_title="Trend Portfolio Simulator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.title("Trend Portfolio Simulator")
st.markdown(
    """
Upload a CSV of manager returns and run a hiring/firing simulation across decades.
Use the sidebar to step through: Upload → Configure → Run → Results → Export.
    """
)
st.info("Open '1_Upload' in the sidebar to get started.")

st.markdown("---")
st.subheader("Quick demo")
st.caption("Explore the app instantly without uploading data.")
if st.button("🎯 Run demo", type="primary"):
    run_one_click_demo()

