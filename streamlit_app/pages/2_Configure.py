import streamlit as st
import pandas as pd
from trend_portfolio_app.policy_engine import PolicyConfig, MetricSpec
from trend_portfolio_app.metrics_extra import AVAILABLE_METRICS

st.title("Configure")

if "returns_df" not in st.session_state:
    st.error("Upload data first on the Upload page.")
    st.stop()

df = st.session_state["returns_df"]
cols = df.columns.tolist()

with st.expander("Simulation window"):
    start = st.date_input("Start date", value=df.index.min().date())
    end = st.date_input("End date", value=df.index.max().date())
    freq = st.selectbox("Review frequency", options=["Monthly", "Quarterly"], index=1)
    lookback_months = st.number_input(
        "In-sample lookback (months)", min_value=12, max_value=240, value=36, step=6
    )
    min_track_months = st.number_input(
        "Minimum track record (months)", min_value=6, max_value=120, value=24, step=6
    )

with st.expander("Benchmark & cash"):
    benchmark = st.selectbox(
        "Benchmark column (optional)", options=["<none>"] + cols, index=0
    )
    cash_rate = st.number_input("Cash rate (annualized, %)", value=0.0, step=0.25)

with st.expander("Selection metrics and weights"):
    metric_names = list(AVAILABLE_METRICS.keys())
    selected = st.multiselect(
        "Metrics", options=metric_names, default=["sharpe", "return_ann", "drawdown"]
    )
    weights = {}
    for m in selected:
        weights[m] = st.number_input(f"Weight for {m}", value=1.0, step=0.5)
    top_k = st.number_input("Hire top-k", min_value=1, max_value=200, value=10, step=1)
    bottom_k = st.number_input(
        "Fire bottom-k", min_value=0, max_value=200, value=0, step=1
    )
    cooldown = st.number_input(
        "Cooldown after firing (months)", min_value=0, max_value=36, value=3, step=1
    )

with st.expander("Constraints"):
    max_active = st.number_input(
        "Max active managers", min_value=1, max_value=999, value=100, step=1
    )
    max_weight = st.slider(
        "Max weight per manager", min_value=0.01, max_value=1.0, value=0.10, step=0.01
    )

policy = PolicyConfig(
    top_k=int(top_k),
    bottom_k=int(bottom_k),
    cooldown_months=int(cooldown),
    min_track_months=int(min_track_months),
    max_active=int(max_active),
    max_weight=float(max_weight),
    metrics=[MetricSpec(name=m, weight=float(weights[m])) for m in selected],
)

st.session_state["sim_config"] = {
    "start": pd.Timestamp(start),
    "end": pd.Timestamp(end),
    "freq": freq.lower(),
    "lookback_months": int(lookback_months),
    "benchmark": None if benchmark == "<none>" else benchmark,
    "cash_rate": float(cash_rate),
    "policy": policy.dict(),
}

st.success("Configuration saved. Proceed to Run.")
