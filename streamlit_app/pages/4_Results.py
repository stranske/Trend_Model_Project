"""Results page for the Streamlit application."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app import state as app_state
from streamlit_app.components import analysis_runner, charts


def _should_auto_render() -> bool:
    """Return True when running inside an active Streamlit session."""

    try:  # pragma: no cover - runtime detection only
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def _analysis_error_messages(error: Exception) -> tuple[str, str | None]:
    """Return a user-friendly summary and optional detail for analysis failures."""

    detail = str(error).strip() or None
    if isinstance(error, ValueError):
        summary = (
            "We couldn't run the analysis with the current data or settings. "
            "Please review the configuration and try again."
        )
    else:
        summary = (
            "We ran into an unexpected problem while generating the report. "
            "Please try again or adjust your configuration."
        )
    return summary, detail


def _current_run_key(model_state: dict[str, Any], benchmark: str | None) -> str:
    fingerprint = st.session_state.get("data_fingerprint", "unknown")
    model_blob = json.dumps(model_state, sort_keys=True, default=str)
    bench = benchmark or "__none__"
    return f"{fingerprint}:{bench}:{model_blob}"


def _prepare_equity_series(returns: pd.Series) -> pd.Series:
    filled = returns.fillna(0.0)
    equity = (1.0 + filled).cumprod()
    equity.name = "Equity"
    return equity


def _prepare_drawdown(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    drawdown.name = "Drawdown"
    return drawdown


def _rolling_sharpe(returns: pd.Series, window: int = 12) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std(ddof=0)
    sharpe = rolling_mean / rolling_std.replace(0.0, np.nan)
    sharpe = sharpe * np.sqrt(12)
    return sharpe.dropna()


def _render_summary(result) -> None:
    metrics = result.metrics
    if metrics is not None and not metrics.empty:
        st.subheader("Summary metrics")
        st.dataframe(metrics)


def _render_charts(result) -> None:
    details = result.details if hasattr(result, "details") else {}
    returns = details.get("portfolio_equal_weight_combined")
    if isinstance(returns, pd.Series) and not returns.empty:
        equity = _prepare_equity_series(returns)
        drawdown = _prepare_drawdown(equity)
        rolling = _rolling_sharpe(returns)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Equity curve")
            st.altair_chart(charts.equity_chart(equity), use_container_width=True)
        with c2:
            st.subheader("Drawdown")
            st.altair_chart(charts.drawdown_chart(drawdown), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Rolling Sharpe (12m)")
            st.altair_chart(
                charts.rolling_sharpe_chart(rolling), use_container_width=True
            )
        with c4:
            turnover = (
                details.get("risk_diagnostics", {}).get("turnover")
                if isinstance(details.get("risk_diagnostics"), dict)
                else None
            )
            st.subheader("Turnover")
            if isinstance(turnover, pd.Series) and not turnover.empty:
                st.altair_chart(
                    charts.turnover_chart(turnover), use_container_width=True
                )
            else:
                st.caption("Turnover data unavailable.")

        exposures = None
        risk_diag = details.get("risk_diagnostics")
        if isinstance(risk_diag, dict):
            exposures = risk_diag.get("final_weights")
        st.subheader("Exposures")
        if exposures is not None:
            st.altair_chart(charts.exposure_chart(exposures), use_container_width=True)
        else:
            st.caption("Exposure breakdown unavailable.")
    else:
        st.info("Run the analysis to see portfolio charts.")


def render_results_page() -> None:
    app_state.initialize_session_state()
    st.title("Results")

    df, _ = app_state.get_uploaded_data()
    if df is None:
        st.error("Load data on the Data page before viewing results.")
        return

    model_state = st.session_state.get("model_state")
    if not isinstance(model_state, dict):
        st.error("Configure the model before generating results.")
        return

    benchmark = st.session_state.get("selected_benchmark")
    run_key = _current_run_key(model_state, benchmark)
    cached_key = st.session_state.get("analysis_result_key")
    result = st.session_state.get("analysis_result") if cached_key == run_key else None

    st.markdown("Run the analysis to generate performance and risk diagnostics.")
    run_clicked = st.button("Run analysis", type="primary")

    if run_clicked or result is None:
        with st.spinner("Running analysis…"):
            try:
                data_hash = st.session_state.get("data_fingerprint")
                result = analysis_runner.run_analysis(
                    df, model_state, benchmark, data_hash=data_hash
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                summary, detail = _analysis_error_messages(exc)
                st.error(summary)
                if detail:
                    st.caption(detail)
                st.session_state["analysis_result"] = None
                st.session_state["analysis_result_key"] = None
                st.session_state["analysis_error"] = {
                    "message": summary,
                    "detail": detail,
                }
                return
        st.session_state["analysis_result"] = result
        st.session_state["analysis_result_key"] = run_key
        st.session_state.pop("analysis_error", None)

    if result is None:
        st.info("Click run analysis to generate a report.")
        return

    fallback = getattr(result, "fallback_info", None)
    if isinstance(fallback, dict):
        st.warning(
            "Weight engine fallback: using equal weights because "
            f"{fallback.get('engine', 'unknown engine')} failed."
        )

    _render_summary(result)
    _render_charts(result)


if _should_auto_render():  # pragma: no cover - Streamlit runtime only
    render_results_page()
