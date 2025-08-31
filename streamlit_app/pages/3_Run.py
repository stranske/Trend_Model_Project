import pandas as pd
import streamlit as st

from trend_analysis.api import run_simulation
from trend_analysis.config import Config

st.title("Run")

if "returns_df" not in st.session_state or "sim_config" not in st.session_state:
    st.error("Upload data and set configuration first.")
    st.stop()

df = st.session_state["returns_df"]
cfg = st.session_state["sim_config"]

# Ensure 'Date' column exists for the pipeline
returns = df.reset_index().rename(columns={df.index.name or "index": "Date"})

progress = st.progress(0, "Running simulation...")
lookback = cfg.get("lookback_months", 0)
start = cfg["start"]
end = cfg["end"]

config = Config(
    version="1",
    data={},
    preprocessing={},
    vol_adjust={"target_vol": cfg.get("risk_target", 1.0)},
    sample_split={
        "in_start": (start - pd.DateOffset(months=lookback)).strftime("%Y-%m"),
        "in_end": (start - pd.DateOffset(months=1)).strftime("%Y-%m"),
        "out_start": start.strftime("%Y-%m"),
        "out_end": end.strftime("%Y-%m"),
    },
    portfolio={},
    metrics={},
    export={},
    run={},
)

result = run_simulation(config, returns)
progress.progress(100)
st.session_state["sim_results"] = result
st.success("Done.")
st.write("Summary:", result.metrics)
