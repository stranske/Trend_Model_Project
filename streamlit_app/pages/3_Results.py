"""Results page presenting charts and diagnostics for the configured model."""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd
import streamlit as st

from app.streamlit import state as app_state
from streamlit_app.components.analysis_runner import (
    ModelSettings,
    build_pipeline_config,
    build_policy_config,
    derive_analysis_window,
    prepare_returns_panel,
)
from streamlit_app.components.charts import bar_chart, category_bar_chart, line_chart
from trend_analysis.api import run_simulation


EPS = 1e-12


def _get_settings() -> ModelSettings | None:
    settings = st.session_state.get("model_settings")
    return settings if isinstance(settings, ModelSettings) else None


def _get_mapping() -> Dict[str, Any] | None:
    mapping = st.session_state.get("model_column_mapping")
    if isinstance(mapping, dict) and mapping.get("return_columns"):
        return mapping
    return None


def _render_metrics(returns: pd.Series) -> None:
    if returns.empty:
        return
    total_return = (1 + returns.fillna(0.0)).prod() - 1
    annualised = (1 + total_return) ** (12 / max(len(returns), 1)) - 1
    equity = (1 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total return", f"{total_return:.1%}")
    c2.metric("Approx. annual return", f"{annualised:.1%}")
    c3.metric("Max drawdown", f"{max_dd:.1%}")


def _plot_time_series(returns: pd.Series) -> None:
    if returns.empty:
        st.info("No portfolio returns available for plotting.")
        return
    returns = returns.sort_index()
    returns.index = pd.to_datetime(returns.index)
    equity = (1 + returns.fillna(0.0)).cumprod()
    equity_df = equity.to_frame("Equity")
    equity_df.index.name = "Date"
    st.altair_chart(
        line_chart(
            equity_df,
            x="Date",
            value_fields=["Equity"],
            title="Equity curve",
            y_title="Equity",
        ),
        use_container_width=True,
    )

    drawdown = equity / equity.cummax() - 1
    dd_df = drawdown.to_frame("Drawdown")
    dd_df.index.name = "Date"
    st.altair_chart(
        line_chart(
            dd_df,
            x="Date",
            value_fields=["Drawdown"],
            title="Drawdown",
            y_title="Drawdown",
            y_format=".1%",
        ),
        use_container_width=True,
    )

    window = 12
    rolling = returns.rolling(window)
    sharpe = rolling.mean() * math.sqrt(12) / (rolling.std(ddof=0) + EPS)
    sharpe_df = sharpe.to_frame("Rolling Sharpe")
    sharpe_df.index.name = "Date"
    st.altair_chart(
        line_chart(
            sharpe_df.dropna(),
            x="Date",
            value_fields=["Rolling Sharpe"],
            title="Rolling Sharpe (12m)",
            y_title="Sharpe",
            y_format=".2f",
        ),
        use_container_width=True,
    )


def _plot_turnover(result, end: pd.Timestamp | None) -> None:
    risk_diag = getattr(result, "details", {}).get("risk_diagnostics") if hasattr(result, "details") else None
    turnover_series = None
    if isinstance(risk_diag, dict):
        turnover = risk_diag.get("turnover")
        if isinstance(turnover, pd.Series) and not turnover.empty:
            idx = pd.to_datetime(turnover.index, errors="coerce")
            if idx.isna().all() and end is not None:
                idx = pd.Index([end] * len(turnover), name="Date")
            turnover_series = pd.Series(turnover.values.astype(float), index=idx, name="Turnover")
        elif isinstance(turnover, (int, float)) and end is not None:
            turnover_series = pd.Series([float(turnover)], index=pd.Index([end], name="Date"), name="Turnover")
    if turnover_series is None:
        st.info("Turnover diagnostics were not produced for this run.")
        return
    turnover_df = turnover_series.to_frame()
    st.altair_chart(
        bar_chart(
            turnover_df,
            x="Date",
            y="Turnover",
            title="Turnover",
            y_title="Turnover",
            y_format=".2f",
        ),
        use_container_width=True,
    )


def _plot_exposures(result) -> None:
    risk_diag = getattr(result, "details", {}).get("risk_diagnostics") if hasattr(result, "details") else None
    exposures = None
    if isinstance(risk_diag, dict):
        final_weights = risk_diag.get("final_weights")
        if isinstance(final_weights, pd.Series):
            exposures = final_weights
        elif isinstance(final_weights, dict):
            exposures = pd.Series(final_weights)
    if exposures is None and hasattr(result, "details"):
        fw = result.details.get("fund_weights")
        if isinstance(fw, dict):
            exposures = pd.Series(fw)
    if exposures is None or exposures.empty:
        st.info("Exposure summary unavailable.")
        return
    exposures = exposures.sort_values(ascending=False)
    top_exposures = exposures.head(15).reset_index()
    top_exposures.columns = ["Manager", "Weight"]
    st.altair_chart(
        category_bar_chart(
            top_exposures,
            category="Manager",
            value="Weight",
            title="Final exposures",
            value_format=".1%",
        ),
        use_container_width=True,
    )


def _run_analysis(df: pd.DataFrame, settings: ModelSettings, mapping: Dict[str, Any]) -> None:
    return_columns = mapping.get("return_columns", [])
    if not return_columns:
        raise ValueError("No return columns selected.")
    benchmark = mapping.get("benchmark_column")
    panel = prepare_returns_panel(
        df,
        date_index_name=df.index.name,
        return_columns=return_columns,
        benchmark_column=benchmark,
    )
    start, end = derive_analysis_window(df.index, lookback_months=settings.lookback_months)
    policy = build_policy_config(settings)
    config = build_pipeline_config(
        settings=settings,
        policy=policy,
        start=start,
        end=end,
        benchmark=benchmark,
    )
    returns = panel.rename(columns={panel.columns[0]: "Date"})
    result = run_simulation(config, returns)
    portfolio = result.details.get("portfolio_equal_weight_combined")
    if isinstance(portfolio, pd.Series):
        st.session_state["portfolio_returns"] = portfolio
    else:
        st.session_state["portfolio_returns"] = pd.Series(dtype=float)
    st.session_state["sim_results"] = result
    st.session_state["analysis_window"] = (start, end)


def main() -> None:
    app_state.initialize_session_state()
    st.title("📊 Results")

    df = st.session_state.get("returns_df")
    settings = _get_settings()
    mapping = _get_mapping()

    if df is None:
        st.error("Load data first on the Data page.")
        st.stop()
    if settings is None or mapping is None:
        st.info("Configure the model before running the analysis.")
        st.stop()

    if st.button("Run analysis", type="primary"):
        with st.spinner("Running analysis…"):
            try:
                _run_analysis(df, settings, mapping)
                st.success("Analysis complete.")
            except Exception as exc:
                st.session_state.pop("sim_results", None)
                st.error(f"Analysis failed: {exc}")
                return

    result = st.session_state.get("sim_results")
    returns = st.session_state.get("portfolio_returns")
    if not hasattr(result, "details") or not isinstance(returns, pd.Series):
        st.info("Run the analysis to view charts.")
        return

    _render_metrics(returns)
    _plot_time_series(returns)
    window = st.session_state.get("analysis_window")
    end = window[1] if isinstance(window, tuple) else None
    _plot_turnover(result, end if isinstance(end, pd.Timestamp) else None)
    _plot_exposures(result)


main()
