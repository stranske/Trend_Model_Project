# --- Streamlit UI ---
import streamlit as st

from streamlit_app.demo import DemoRunError, run_one_click_demo

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

st.divider()
st.subheader("Try the app instantly")
st.write(
    "Load the built-in dataset, apply the Balanced preset, and jump straight to the results and export pages."
)

if st.button("🎬 Run demo", type="primary"):
    with st.spinner("Loading demo dataset and running the analysis..."):
        try:
            run_one_click_demo(st.session_state)
        except DemoRunError as exc:
            st.error(f"Demo run failed: {exc}")
        else:
            st.success("Demo complete! Opening results…")
            if hasattr(st, "switch_page"):
                try:
                    st.switch_page("pages/4_Results.py")
                except Exception:  # pragma: no cover - defensive UI fallback
                    st.info("Use the sidebar to open the Results page.")
