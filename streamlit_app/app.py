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
Welcome! Start on **Data** to load a CSV or try the bundled sample dataset,
move to **Model** to choose presets and portfolio settings, then open
**Results** for performance charts. Use **Export** to download outputs when
you're happy with the configuration.
    """
)
st.info("Open '1_Data' in the sidebar to get started.")

st.markdown("---")
st.subheader("Quick start")
if st.button("🚀 Run demo", type="primary"):
    with st.spinner("Loading demo data and running the analysis..."):
        success = run_one_click_demo()
    if success:
        st.success("Demo ready! Redirecting to Results…")
        st.switch_page("pages/4_Results.py")
