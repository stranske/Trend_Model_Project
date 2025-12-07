# --- Streamlit UI ---
import streamlit as st

from streamlit_app.components.demo_runner import run_one_click_demo

st.set_page_config(
    page_title="Portfolio Simulator",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.title("Portfolio Simulator")
st.markdown(
    """
Welcome! This app analyzes trend-following fund portfolios with volatility adjustment.

**Quick Start Options:**
- **Run Demo** - Load sample data and run analysis with default settings
- **Configure First** - Load sample data and customize settings before running

**Manual Flow:**
1. **Data** - Load your own CSV or use the bundled sample dataset
2. **Model** - Configure lookback periods, selection criteria, and portfolio settings
3. **Results** - View performance charts and metrics
    """
)

st.markdown("---")
st.subheader("Quick Start")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Run Demo with Defaults", type="primary", use_container_width=True):
        with st.spinner(
            "Loading demo data and running analysis with default settings..."
        ):
            success = run_one_click_demo()
        if success:
            st.success("Demo complete! Viewing results...")
            st.switch_page("pages/3_Results.py")
        else:
            st.error("Demo failed. Check the error message above.")

with col2:
    if st.button("⚙️ Configure Demo First", use_container_width=True):
        # Load demo data but go to Model page for configuration
        with st.spinner("Loading demo data..."):
            from streamlit_app.components.demo_runner import _load_demo_returns
            from streamlit_app.components.data_cache import cache_key_for_frame
            from streamlit_app import state as app_state
            from trend_portfolio_app.data_schema import infer_benchmarks

            try:
                df, meta = _load_demo_returns()
                # Store in session state so Model page can use it
                app_state.initialize_session_state()
                st.session_state["returns_df"] = df
                st.session_state["schema_meta"] = meta
                st.session_state["benchmark_candidates"] = infer_benchmarks(
                    list(df.columns)
                )
                st.session_state["data_loaded_key"] = f"demo::{cache_key_for_frame(df)}"
                st.success("Demo data loaded! Redirecting to Model configuration...")
                st.switch_page("pages/2_Model.py")
            except Exception as exc:
                st.error(f"Failed to load demo data: {exc}")

st.markdown("---")
st.caption("Or select a page from the sidebar to start manually.")
