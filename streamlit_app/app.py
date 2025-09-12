# --- Streamlit UI ---
import streamlit as st

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

# --- FastAPI health probe ---
try:
    from fastapi import FastAPI, Response
except ImportError:
    app = None
else:
    app = FastAPI()

    @app.get("/health", response_class=Response)
    async def health():
        return Response(content="OK", media_type="text/plain")
