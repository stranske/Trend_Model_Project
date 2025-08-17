import streamlit as st
from trend_portfolio_app.sim_runner import Simulator
from trend_portfolio_app.policy_engine import PolicyConfig

st.title("Run")

if "returns_df" not in st.session_state or "sim_config" not in st.session_state:
    st.error("Upload data and set configuration first.")
    st.stop()

df = st.session_state["returns_df"]
cfg = st.session_state["sim_config"]
policy = PolicyConfig(**cfg["policy"])

sim = Simulator(df, benchmark_col=cfg["benchmark"], cash_rate_annual=cfg["cash_rate"])
progress = st.progress(0, "Running simulation...")

results = sim.run(
    start=cfg["start"],
    end=cfg["end"],
    freq=cfg["freq"],
    lookback_months=cfg["lookback_months"],
    policy=policy,
    rebalance=cfg.get("rebalance", {}),
    progress_cb=lambda i, n: (
        progress.progress(int(100 * i / n), text=f"Running period {i}/{n}") and None
    ),
)

st.session_state["sim_results"] = results
st.success("Done.")
st.write("Summary:", results.summary())
