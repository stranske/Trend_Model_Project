# --- Streamlit UI ---
import streamlit as st

from streamlit_app.components.demo_runner import run_one_click_demo

st.set_page_config(
    page_title="Trend Portfolio Studio",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Welcome to Trend Portfolio Studio")
st.markdown(
    """
The Streamlit app is now the **front door** to the trend portfolio toolkit.
Work from left to right – **Data → Model → Results** – and the app will guide you
from loading returns data through configuring the simulation to reviewing the
report.
    """
)

with st.container():
    st.markdown("### First time here?")
    st.write(
        "Start on the **Data** page. A curated sample dataset loads automatically "
        "so you can explore the workflow before uploading your own CSV."
    )

st.markdown("---")
st.markdown("### Try the guided demo")
st.write(
    "Prefer to see the full experience immediately? Run the demo to populate the "
    "Model and Results pages with a ready-made configuration."
)

if st.button("🚀 Run guided demo", type="primary"):
    with st.spinner("Loading sample data and preparing results…"):
        success = run_one_click_demo()
    if success:
        st.success(
            "Demo ready! Review the outcomes on the Results page or customise the "
            "model next."
        )
        st.switch_page("pages/3_Results.py")
