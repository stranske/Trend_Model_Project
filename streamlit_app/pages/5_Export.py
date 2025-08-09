import streamlit as st
from trend_portfolio_app.io_utils import export_bundle

st.title("Export")

if "sim_results" not in st.session_state or "sim_config" not in st.session_state:
    st.error("Run a simulation first.")
    st.stop()

res = st.session_state["sim_results"]
cfg = st.session_state["sim_config"]

path = export_bundle(res, cfg)
st.success(f"Bundle created: {path}")
with open(path, "rb") as f:
    st.download_button("Download bundle", f, file_name=path.split("/")[-1])
