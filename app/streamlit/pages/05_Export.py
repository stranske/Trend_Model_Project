"""Export bundle download page for the Streamlit app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from trend_analysis.export.bundle import export_bundle

st.title("Export")

if "sim_results" not in st.session_state or "sim_config" not in st.session_state:
    st.error("Run a simulation first.")
    st.stop()

run = st.session_state["sim_results"]
# attach config and seed if available
setattr(run, "config", st.session_state.get("sim_config", {}))
setattr(run, "seed", st.session_state.get("seed", None))

# Create bundle on demand and offer download
if st.button("Create bundle"):
    tmpdir = Path(tempfile.mkdtemp())
    zip_path = tmpdir / "analysis_bundle.zip"
    export_bundle(run, zip_path)
    with open(zip_path, "rb") as f:
        st.download_button("Download bundle", f, file_name=zip_path.name)
