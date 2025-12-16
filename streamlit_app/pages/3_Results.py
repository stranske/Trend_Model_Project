"""Results page for the Streamlit application."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app import state as app_state
from streamlit_app.components import analysis_runner, charts

# =============================================================================
# Formatting Helpers
# =============================================================================


def _fmt_pct(x: float, decimals: int = 1) -> str:
    """Format as percentage with specified decimals."""
    if pd.isna(x) or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def _fmt_ratio(x: float) -> str:
    """Format ratios (Sharpe, Sortino) to 2 decimal places."""
    if pd.isna(x) or not np.isfinite(x):
        return "—"
    return f"{x:.2f}"


# =============================================================================
# Utility Functions
# =============================================================================


def _should_auto_render() -> bool:
    """Return True when running inside an active Streamlit session."""
    try:
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


def _diagnostic_message(result: Any) -> tuple[str | None, str | None]:
    """Extract a human-readable failure message from a RunResult-like object."""
    details = getattr(result, "details", {}) or {}
    diag = getattr(result, "diagnostic", None)

    if isinstance(details, dict) and details.get("error"):
        return "Analysis failed to produce results.", str(details.get("error"))

    if diag is not None:
        message = getattr(diag, "message", None)
        code = getattr(diag, "reason_code", None)
        if message:
            summary = (
                f"Analysis did not produce results ({code})."
                if code
                else "Analysis did not produce results."
            )
            return summary, str(message)

    return None, None


def _coerce_weight_mapping(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, pd.Series):
        items = raw.items()
    else:
        return {}
    out: dict[str, float] = {}
    for k, v in items:
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    total = sum(out.values())
    # Heuristic: if weights look like percents, scale to fractions.
    if total > 2.0:
        out = {k: v / 100.0 for k, v in out.items()}
    return out


def _compute_weighted_portfolio_returns(
    out_df: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    if out_df.empty or not weights:
        return pd.Series(dtype=float)
    cols = [str(c) for c in out_df.columns]
    w = np.array([weights.get(c, 0.0) for c in cols], dtype=float)
    # If the portfolio weights don't sum to 1 (due to rounding/constraints),
    # renormalise so the portfolio return is interpretable.
    w_sum = float(np.sum(np.abs(w)))
    if w_sum > 0 and abs(w_sum - 1.0) > 1e-6:
        w = w / w_sum
    port = out_df.mul(w, axis=1).sum(axis=1)
    port.name = "Portfolio"
    return port


def _portfolio_series_from_details(details: dict[str, Any]) -> pd.Series:
    """Best-effort portfolio series builder for UI rendering.

    Uses out-of-sample returns and the weights actually applied per period.
    """

    period_results = details.get("period_results")
    if isinstance(period_results, list) and period_results:
        parts: list[pd.Series] = []
        for res in period_results:
            if not isinstance(res, dict):
                continue
            out_df = res.get("out_sample_scaled")
            if not isinstance(out_df, pd.DataFrame) or out_df.empty:
                continue
            weights_raw = res.get("fund_weights") or res.get("ew_weights")
            weights = _coerce_weight_mapping(weights_raw)
            series = _compute_weighted_portfolio_returns(out_df, weights)
            if not series.empty:
                parts.append(series)
        if parts:
            combined = pd.concat(parts).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]
            return combined

    out_df = details.get("out_sample_scaled")
    if isinstance(out_df, pd.DataFrame) and not out_df.empty:
        weights_raw = details.get("fund_weights") or details.get("ew_weights")
        weights = _coerce_weight_mapping(weights_raw)
        return _compute_weighted_portfolio_returns(out_df, weights)

    return pd.Series(dtype=float)


def _current_run_key(model_state: dict[str, Any], benchmark: str | None) -> str:
    fingerprint = st.session_state.get("data_fingerprint", "unknown")
    model_blob = json.dumps(model_state, sort_keys=True, default=str)
    bench = benchmark or "__none__"
    applied_funds = st.session_state.get("analysis_fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = st.session_state.get("fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = []

    selected_rf = st.session_state.get("selected_risk_free")
    info_ratio_benchmark = (
        model_state.get("info_ratio_benchmark")
        if isinstance(model_state, dict)
        else None
    )
    prohibited = {selected_rf, benchmark, info_ratio_benchmark} - {None}
    sanitized_funds = [c for c in applied_funds if c not in prohibited]

    funds_blob = json.dumps(list(sanitized_funds), sort_keys=False, default=str)
    funds_hash = hashlib.sha256(funds_blob.encode("utf-8")).hexdigest()[:12]
    return f"{fingerprint}:{bench}:{funds_hash}:{model_blob}"


def _prepare_equity_series(returns: pd.Series) -> pd.Series:
    filled = returns.fillna(0.0)
    equity = (1.0 + filled).cumprod()
    equity.name = "Cumulative Return"
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


def _get_simulation_date_range(result) -> tuple[str | None, str | None]:
    """Extract the simulation start and end dates from multi-period results."""
    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", [])

    if not period_results:
        return None, None

    # Get first period's out-sample start and last period's out-sample end
    first_period = period_results[0].get("period", ("", "", "", ""))
    last_period = period_results[-1].get("period", ("", "", "", ""))

    sim_start = first_period[2] if len(first_period) > 2 else None
    sim_end = last_period[3] if len(last_period) > 3 else None

    return sim_start, sim_end


def _get_portfolio_funds_by_period(result) -> dict[str, set[str]]:
    """Get the set of funds in portfolio at each period."""
    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", [])

    portfolio_by_period: dict[str, set[str]] = {}

    eps = 1e-12

    for res in period_results:
        period = res.get("period", ("", "", "", ""))
        out_start = period[2] if len(period) > 2 else ""

        # Prefer realised positive weights (actual portfolio) over selected_funds.
        weights = res.get("fund_weights") or res.get("ew_weights") or {}
        holdings: set[str] = set()
        if isinstance(weights, dict) and weights:
            for name, value in weights.items():
                try:
                    if float(value or 0.0) > eps and str(name).strip():
                        holdings.add(str(name))
                except Exception:
                    continue

        if not holdings:
            selected = res.get("selected_funds", [])
            if isinstance(selected, (list, tuple)) and selected:
                holdings = {str(x) for x in selected if str(x).strip()}

        portfolio_by_period[out_start] = holdings

    return portfolio_by_period


def _get_final_portfolio(result) -> set[str]:
    """Get the funds in the final portfolio."""
    portfolio_by_period = _get_portfolio_funds_by_period(result)
    if not portfolio_by_period:
        return set()
    last_date = max(portfolio_by_period.keys())
    return portfolio_by_period[last_date]


# =============================================================================
# Trailing Period Stats (1, 3, 5, 10, LOF)
# =============================================================================


def _compute_trailing_stats(returns: pd.Series) -> pd.DataFrame:
    """Compute trailing period statistics for 1, 3, 5, 10 years and life-of-fund."""
    if returns.empty:
        return pd.DataFrame()

    periods = {
        "1Y": 12,
        "3Y": 36,
        "5Y": 60,
        "10Y": 120,
        "LOF": len(returns),
    }

    stats_data = []
    for label, months in periods.items():
        if months > len(returns):
            if label != "LOF":
                continue
            months = len(returns)

        subset = returns.iloc[-months:]
        if subset.empty:
            continue

        total_return = (1 + subset).prod() - 1
        years = months / 12
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        vol = subset.std() * np.sqrt(12)
        sharpe = (cagr / vol) if vol > 0 else 0

        downside = subset[subset < 0]
        downside_std = downside.std() * np.sqrt(12) if len(downside) > 0 else 0
        sortino = (cagr / downside_std) if downside_std > 0 else 0

        equity = (1 + subset).cumprod()
        rolling_max = equity.cummax()
        drawdown = equity / rolling_max - 1
        max_dd = drawdown.min()

        stats_data.append(
            {
                "Period": label,
                "CAGR": cagr,
                "Volatility": vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Max DD": max_dd,
            }
        )

    if not stats_data:
        return pd.DataFrame()

    return pd.DataFrame(stats_data).set_index("Period")


def _render_trailing_stats(returns: pd.Series) -> None:
    """Render trailing period statistics table."""
    stats_df = _compute_trailing_stats(returns)
    if stats_df.empty:
        st.caption("Insufficient data for trailing statistics.")
        return

    formatted = stats_df.copy()
    for col in ["CAGR", "Volatility", "Max DD"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: _fmt_pct(x, 1))
    for col in ["Sharpe", "Sortino"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(_fmt_ratio)

    st.dataframe(formatted, use_container_width=True)


# =============================================================================
# Manager Changes / Hiring & Firing Decisions
# =============================================================================


def _extract_manager_changes(result) -> pd.DataFrame:
    """Extract manager changes that occurred during simulation period.

    Includes seed entries as 'Initial Portfolio' for the first period.
    """
    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", [])

    if not period_results:
        return pd.DataFrame()

    # Compute net hires/terminations per period from the *actual* holdings.
    # We prefer `selected_funds` / non-zero weights because the engine event log
    # can be incomplete or contain intermediate actions that do not reflect the
    # final portfolio for the period.
    all_changes: list[dict[str, str]] = []
    prev_holdings: set[str] | None = None

    def _last_event_for(
        events: list[dict[str, Any]], manager: str, action: str
    ) -> dict[str, Any] | None:
        for ev in reversed(events):
            if ev.get("manager") == manager and ev.get("action") == action:
                return ev
        return None

    eps = 1e-12

    for res in period_results:
        period = res.get("period", ("", "", "", ""))
        out_start = period[2] if len(period) > 2 else ""
        raw_changes = res.get("manager_changes", [])
        changes = [
            c
            for c in (raw_changes or [])
            if isinstance(c, dict) and str(c.get("manager", "")).strip()
        ]

        # Prefer realised positive weights (actual portfolio) over selected_funds.
        weights = res.get("fund_weights") or res.get("ew_weights") or {}
        holdings: set[str] = set()
        if isinstance(weights, dict) and weights:
            for name, value in weights.items():
                try:
                    if float(value or 0.0) > eps and str(name).strip():
                        holdings.add(str(name))
                except Exception:
                    continue

        if not holdings:
            selected = res.get("selected_funds", [])
            if isinstance(selected, (list, tuple)) and selected:
                holdings = {str(x) for x in selected if str(x).strip()}

        if prev_holdings is None:
            for manager in sorted(holdings):
                all_changes.append(
                    {
                        "Date": out_start,
                        "Action": "Initial",
                        "Manager": manager,
                        "Reason": "Initial portfolio selection",
                    }
                )
            prev_holdings = holdings
            continue

        hired = sorted(holdings - prev_holdings)
        terminated = sorted(prev_holdings - holdings)

        for manager in hired:
            ev = _last_event_for(changes, manager, "added")
            reason = str((ev or {}).get("reason", ""))
            detail = str((ev or {}).get("detail", ""))
            all_changes.append(
                {
                    "Date": out_start,
                    "Action": "Hired",
                    "Manager": manager,
                    "Reason": _format_change_reason(reason, detail, "added"),
                }
            )

        for manager in terminated:
            ev = _last_event_for(changes, manager, "dropped")
            reason = str((ev or {}).get("reason", ""))
            detail = str((ev or {}).get("detail", ""))
            all_changes.append(
                {
                    "Date": out_start,
                    "Action": "Terminated",
                    "Manager": manager,
                    "Reason": _format_change_reason(reason, detail, "dropped"),
                }
            )

        prev_holdings = holdings

    return pd.DataFrame(all_changes) if all_changes else pd.DataFrame()


def _format_change_reason(reason: str, detail: str, action: str) -> str:
    """Format the reason into a readable explanation."""
    reason_map = {
        "z_exit": "Performance below threshold",
        "z_entry": "Performance above threshold",
        "rebalance": "Portfolio rebalance",
        "one_per_firm": "Firm concentration limit",
        "low_weight_strikes": "Persistent underweight",
        "replacement": "Replaced underperformer",
        "reseat": "Portfolio reconstruction",
    }

    base_reason = reason_map.get(reason, reason.replace("_", " ").title())

    # Extract z-score if available
    if "zscore=" in detail:
        try:
            z_val = float(detail.split("zscore=")[1].split()[0])
            if action == "dropped":
                return f"{base_reason} (z={z_val:.2f})"
            else:
                return f"{base_reason} (z={z_val:.2f})"
        except (ValueError, IndexError):
            pass

    return base_reason


def _render_manager_changes(result) -> None:
    """Render manager hiring/firing decisions table."""
    changes_df = _extract_manager_changes(result)

    if changes_df.empty:
        st.caption("No manager changes during simulation period.")
        return

    initial = len(changes_df[changes_df["Action"] == "Initial"])
    hired = len(changes_df[changes_df["Action"] == "Hired"])
    terminated = len(changes_df[changes_df["Action"] == "Terminated"])

    summary_parts = []
    if initial > 0:
        summary_parts.append(f"{initial} initial")
    if hired > 0:
        summary_parts.append(f"{hired} hired")
    if terminated > 0:
        summary_parts.append(f"{terminated} terminated")

    st.caption(f"Total: {len(changes_df)} ({', '.join(summary_parts)})")

    def highlight_action(row):
        if row["Action"] == "Initial":
            return ["background-color: #cce5ff"] * len(row)  # Blue for initial
        elif row["Action"] == "Hired":
            return ["background-color: #d4edda"] * len(row)  # Green for hired
        elif row["Action"] == "Terminated":
            return ["background-color: #f8d7da"] * len(row)  # Red for terminated
        return [""] * len(row)

    styled = changes_df.style.apply(highlight_action, axis=1)
    st.dataframe(styled, use_container_width=True, height=300)


# =============================================================================
# Fund Holding Periods with Risk Stats
# =============================================================================


def _compute_fund_stats(returns: pd.Series) -> dict[str, float]:
    """Compute risk statistics for a fund's return series."""
    if returns.empty or len(returns) < 2:
        return {}

    # Annualized return
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 12
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Volatility
    vol = returns.std() * np.sqrt(12)

    # Sharpe
    sharpe = (cagr / vol) if vol > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(12) if len(downside) > 0 else 0
    sortino = (cagr / downside_std) if downside_std > 0 else 0

    # Max drawdown
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
    }


def _compute_fund_holding_periods(result) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute holding periods and stats for each fund.

    Returns:
        tuple: (summary_df, risk_stats_df)
    """
    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", [])

    if not period_results:
        return pd.DataFrame(), pd.DataFrame()

    # Get frequency to properly calculate years
    # Default to annual (A) if not specified
    model_state = st.session_state.get("model_state", {})
    frequency = model_state.get("multi_period_frequency", "A") or "A"

    # Determine months per period based on frequency
    freq_to_months = {
        "A": 12,  # Annual = 12 months
        "S": 6,  # Semi-annual = 6 months
        "Q": 3,  # Quarterly = 3 months
        "M": 1,  # Monthly = 1 month
    }
    months_per_period = freq_to_months.get(frequency, 12)

    # Get simulation start date for seed funds
    sim_start, _ = _get_simulation_date_range(result)

    # Track fund entry/exit and accumulate returns
    fund_tenures: dict[str, dict] = {}
    fund_returns: dict[str, list[float]] = {}

    for res in period_results:
        period = res.get("period", ("", "", "", ""))
        out_start = period[2] if len(period) > 2 else ""
        changes = res.get("manager_changes", [])
        out_df = res.get("out_sample_scaled")

        for change in changes:
            manager = change.get("manager", "")
            action = change.get("action", "")
            reason = change.get("reason", "")

            if action == "added":
                if manager not in fund_tenures:
                    # Use simulation start for seed funds, out_start for others
                    entry_date = sim_start if reason == "seed" else out_start
                    fund_tenures[manager] = {
                        "Manager": manager,
                        "Entry Date": entry_date,
                        "Exit Date": None,
                        "periods_held": 0,
                        "is_seed": reason == "seed",
                    }
                    fund_returns[manager] = []
            elif action == "dropped":
                if (
                    manager in fund_tenures
                    and fund_tenures[manager]["Exit Date"] is None
                ):
                    fund_tenures[manager]["Exit Date"] = out_start

        # Accumulate returns for funds held this period
        for manager in fund_tenures:
            tenure = fund_tenures[manager]
            if tenure["Exit Date"] is None or tenure["Exit Date"] >= out_start:
                tenure["periods_held"] += 1
                if isinstance(out_df, pd.DataFrame) and manager in out_df.columns:
                    fund_returns[manager].extend(out_df[manager].dropna().tolist())

    # Build summary and risk stats DataFrames
    summary_rows = []
    risk_rows = []

    for manager, tenure in fund_tenures.items():
        periods = tenure["periods_held"]
        # Calculate years based on frequency (periods * months_per_period / 12)
        years_held = (periods * months_per_period) / 12

        summary_rows.append(
            {
                "Manager": manager,
                "Years Held": years_held,
                "Entry": tenure["Entry Date"] or "",
                "Exit": tenure["Exit Date"] or "Current",
            }
        )

        # Compute risk stats if we have returns
        if manager in fund_returns and fund_returns[manager]:
            returns_series = pd.Series(fund_returns[manager])
            stats = _compute_fund_stats(returns_series)
            risk_rows.append({"Manager": manager, "Years": years_held, **stats})

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    risk_df = pd.DataFrame(risk_rows) if risk_rows else pd.DataFrame()

    if not summary_df.empty:
        summary_df = summary_df.sort_values("Entry")
    if not risk_df.empty:
        risk_df = risk_df.sort_values("Years", ascending=False)

    return summary_df, risk_df


def _render_fund_holdings(result) -> None:
    """Render fund holding periods and risk statistics."""
    summary_df, risk_df = _compute_fund_holding_periods(result)

    if summary_df.empty:
        st.caption("No fund holding data available.")
        return

    # Format summary
    display_summary = summary_df.copy()
    display_summary["Years Held"] = display_summary["Years Held"].apply(
        lambda x: f"{x:.1f}"
    )

    def highlight_current(row):
        if row["Exit"] == "Current":
            return ["background-color: #cce5ff"] * len(row)
        return [""] * len(row)

    st.markdown("**Holding Summary**")
    styled = display_summary.style.apply(highlight_current, axis=1)
    st.dataframe(styled, use_container_width=True, height=200)

    # Risk stats section
    if not risk_df.empty:
        st.markdown("**Out-of-Sample Risk Statistics**")
        display_risk = risk_df.copy()
        display_risk["Years"] = display_risk["Years"].apply(lambda x: f"{x:.1f}")
        for col in ["CAGR", "Vol", "Max DD"]:
            if col in display_risk.columns:
                display_risk[col] = display_risk[col].apply(lambda x: _fmt_pct(x, 1))
        for col in ["Sharpe", "Sortino"]:
            if col in display_risk.columns:
                display_risk[col] = display_risk[col].apply(_fmt_ratio)

        st.dataframe(display_risk, use_container_width=True, height=200)


# =============================================================================
# Period-by-Period Reproducibility Section
# =============================================================================


def _get_selection_config(result) -> dict[str, Any]:
    """Extract selection configuration parameters for display."""
    # Try to get from model_state in session
    model_state = st.session_state.get("model_state", {})

    # Portfolio sizing source of truth (multi-period runs): mp_min_funds/mp_max_funds.
    # Avoid legacy/duplicate sizing knobs here.
    max_funds = int(
        model_state.get("mp_max_funds") or model_state.get("selection_count") or 10
    )
    min_funds = int(model_state.get("mp_min_funds") or 0)

    config = {
        "target_n": int(model_state.get("selection_count") or 8),
        "z_entry_soft": model_state.get("z_entry_soft", 1.0),
        "z_exit_soft": model_state.get("z_exit_soft", -1.0),
        "min_weight": model_state.get("min_weight", 0.05),
        "max_weight": model_state.get("max_weight", 0.2),
        "min_funds": min_funds,
        "max_funds": max_funds,
        "selection_metric": model_state.get("selection_metric", "Sharpe"),
        "weighting_scheme": model_state.get("weighting_scheme", "equal"),
        "lookback_periods": model_state.get("lookback_periods", 3),
        "evaluation_periods": model_state.get("evaluation_periods", 1),
        # In the app, the user-facing "rebalance frequency" is the portfolio
        # rebalance frequency (e.g., quarterly) rather than the multi-period
        # scheduling cadence.
        "frequency": model_state.get("rebalance_freq")
        or model_state.get("multi_period_frequency", "A"),
    }

    return config


def _render_selection_criteria(result) -> None:
    """Render the selection criteria used in the simulation."""
    config = _get_selection_config(result)

    freq_labels = {"A": "Annual", "S": "Semi-Annual", "Q": "Quarterly", "M": "Monthly"}
    freq = freq_labels.get(config["frequency"], config["frequency"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Selection Parameters**")
        st.markdown(f"- Target portfolio size: **{config['target_n']}** funds")
        if config.get("min_funds"):
            st.markdown(f"- Min funds allowed: **{config['min_funds']}**")
        st.markdown(f"- Max funds allowed: **{config['max_funds']}**")
        st.markdown(f"- Selection metric: **{config['selection_metric']}**")
        st.markdown(f"- Weighting: **{config['weighting_scheme']}**")

    with col2:
        st.markdown("**Z-Score Thresholds**")
        st.markdown(f"- Entry threshold: **z ≥ {config['z_entry_soft']:.2f}**")
        st.markdown(f"- Exit threshold: **z ≤ {config['z_exit_soft']:.2f}**")
        st.markdown("*(Higher z-score = better relative performance)*")

    with col3:
        st.markdown("**Time Parameters**")
        st.markdown(f"- Rebalance frequency: **{freq}**")
        st.markdown(f"- In-sample (lookback): **{config['lookback_periods']}** periods")
        st.markdown(
            f"- Out-of-sample (eval): **{config['evaluation_periods']}** period(s)"
        )


def _build_period_detail(res: dict[str, Any], period_num: int) -> dict[str, Any]:
    """Build detailed data for a single period."""
    period = res.get("period", ("", "", "", ""))
    in_start = period[0] if len(period) > 0 else ""
    in_end = period[1] if len(period) > 1 else ""
    out_start = period[2] if len(period) > 2 else ""
    out_end = period[3] if len(period) > 3 else ""

    # Score frame contains in-sample metrics for all candidates
    score_frame = res.get("score_frame")
    if score_frame is None:
        score_frame = pd.DataFrame()

    # Selected funds for this period
    selected_funds = res.get("selected_funds", [])

    # Weights applied
    fund_weights = res.get("fund_weights", {})
    ew_weights = res.get("ew_weights", {})

    # In-sample and out-sample stats
    in_sample_stats = res.get("in_sample_stats", {})
    out_sample_stats = res.get("out_sample_stats", {})

    # Manager changes
    changes = res.get("manager_changes", [])

    # Out-of-sample returns
    out_sample_scaled = res.get("out_sample_scaled")
    if out_sample_scaled is None:
        out_sample_scaled = pd.DataFrame()

    # Portfolio stats for this period
    out_user_stats = res.get("out_user_stats")
    out_ew_stats = res.get("out_ew_stats")

    return {
        "period_num": period_num,
        "in_start": in_start,
        "in_end": in_end,
        "out_start": out_start,
        "out_end": out_end,
        "score_frame": score_frame,
        "selected_funds": selected_funds,
        "fund_weights": fund_weights,
        "ew_weights": ew_weights,
        "in_sample_stats": in_sample_stats,
        "out_sample_stats": out_sample_stats,
        "changes": changes,
        "out_sample_scaled": out_sample_scaled,
        "out_user_stats": out_user_stats,
        "out_ew_stats": out_ew_stats,
    }


def _render_single_period(period_data: dict[str, Any]) -> None:
    """Render detailed view for a single period."""
    pn = period_data["period_num"]

    st.markdown(
        f"### Period {pn}: {period_data['out_start']} to {period_data['out_end']}"
    )
    st.caption(
        f"In-sample window: {period_data['in_start']} to {period_data['in_end']}"
    )

    tabs = st.tabs(
        ["📊 In-Sample Metrics", "✅ Selection", "📈 Out-of-Sample", "💰 Period Return"]
    )

    with tabs[0]:
        # In-sample metrics (score frame)
        score_frame = period_data["score_frame"]
        if isinstance(score_frame, pd.DataFrame) and not score_frame.empty:
            # Sort by primary metric (usually Sharpe) descending
            display_cols = [c for c in score_frame.columns if c not in ["zscore"]]
            if "zscore" in score_frame.columns:
                display_cols.append("zscore")

            sf_display = score_frame[display_cols].copy()

            # Highlight selected funds
            selected = set(period_data["selected_funds"])

            def highlight_selected(row):
                if row.name in selected:
                    return ["background-color: #d4edda"] * len(row)
                return [""] * len(row)

            # Sort by zscore if available, else by first column
            sort_col = (
                "zscore" if "zscore" in sf_display.columns else sf_display.columns[0]
            )
            sf_sorted = sf_display.sort_values(sort_col, ascending=False)

            # Format numeric columns
            for col in sf_sorted.columns:
                if col == "zscore":
                    sf_sorted[col] = sf_sorted[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                    )
                elif col in ["Sharpe", "Sortino", "InformationRatio"]:
                    sf_sorted[col] = sf_sorted[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                    )
                elif col in [
                    "AnnualReturn",
                    "Volatility",
                    "MaxDrawdown",
                    "CAGR",
                    "Vol",
                ]:
                    sf_sorted[col] = sf_sorted[col].apply(
                        lambda x: _fmt_pct(x, 1) if pd.notna(x) else "—"
                    )

            st.markdown(
                "**All candidates ranked by in-sample metrics** (green = selected)"
            )
            styled = sf_sorted.style.apply(highlight_selected, axis=1)
            st.dataframe(styled, use_container_width=True, height=300)
        else:
            st.caption("No in-sample metrics available for this period.")

    with tabs[1]:
        # Selection decisions
        changes = period_data["changes"]
        if changes:
            st.markdown("**Selection Decisions for This Period**")

            added = [c for c in changes if c.get("action") == "added"]
            dropped = [c for c in changes if c.get("action") == "dropped"]

            if added:
                st.markdown("**Added:**")
                for c in added:
                    reason = c.get("reason", "")
                    detail = c.get("detail", "")
                    st.markdown(
                        f"- ✅ **{c.get('manager', '')}** — {_format_change_reason(reason, detail, 'added')}"
                    )

            if dropped:
                st.markdown("**Dropped:**")
                for c in dropped:
                    reason = c.get("reason", "")
                    detail = c.get("detail", "")
                    st.markdown(
                        f"- ❌ **{c.get('manager', '')}** — {_format_change_reason(reason, detail, 'dropped')}"
                    )
        else:
            st.caption("No changes this period (portfolio unchanged).")

        # Show final portfolio weights
        weights = period_data["fund_weights"]
        if weights:
            st.markdown("**Portfolio Weights Applied:**")
            eps = 1e-12
            w_df = pd.DataFrame(
                [
                    {"Fund": k, "Weight": f"{v*100:.1f}%"}
                    for k, v in sorted(weights.items(), key=lambda x: -x[1])
                    if float(v or 0.0) > eps
                ]
            )
            st.dataframe(w_df, use_container_width=True, hide_index=True, height=200)

    with tabs[2]:
        # Out-of-sample returns
        out_df = period_data["out_sample_scaled"]
        selected = period_data["selected_funds"]

        if isinstance(out_df, pd.DataFrame) and not out_df.empty and selected:
            cols_to_show = [c for c in selected if c in out_df.columns]
            if cols_to_show:
                st.markdown("**Out-of-Sample Monthly Returns (Volatility-Adjusted)**")

                try:
                    out_start = str(period_data.get("out_start") or "")[:7]
                    out_end = str(period_data.get("out_end") or "")[:7]
                    if out_start and out_end:
                        expected = pd.period_range(out_start, out_end, freq="M")
                        expected_labels = [str(p) for p in expected]
                        actual_labels = sorted(
                            {
                                str(p)
                                for p in pd.to_datetime(out_df.index)
                                .to_period("M")
                                .tolist()
                            }
                        )
                        missing = [
                            m for m in expected_labels if m not in set(actual_labels)
                        ]
                        if missing:
                            st.warning(
                                "Missing months inside this out-of-sample window: "
                                + ", ".join(missing)
                            )
                except Exception:
                    pass

                # Show summary stats
                out_stats_map = period_data.get("out_sample_stats")
                if not isinstance(out_stats_map, dict):
                    out_stats_map = {}

                oos_summary = []
                for col in cols_to_show:
                    returns = out_df[col].dropna()
                    if len(returns) > 0:
                        total_ret = (1 + returns).prod() - 1

                        fallback_stats = _compute_fund_stats(returns)

                        stats_obj = out_stats_map.get(col)
                        stats_map = (
                            stats_obj
                            if isinstance(stats_obj, dict)
                            else (vars(stats_obj) if stats_obj is not None else {})
                        )

                        vol = (
                            stats_map.get("vol")
                            if stats_map.get("vol") is not None
                            else fallback_stats.get("Vol")
                        )
                        cagr = (
                            stats_map.get("cagr")
                            if stats_map.get("cagr") is not None
                            else fallback_stats.get("CAGR")
                        )
                        sharpe = (
                            stats_map.get("sharpe")
                            if stats_map.get("sharpe") is not None
                            else fallback_stats.get("Sharpe")
                        )
                        sortino = (
                            stats_map.get("sortino")
                            if stats_map.get("sortino") is not None
                            else fallback_stats.get("Sortino")
                        )
                        info_ratio = stats_map.get("information_ratio")
                        max_dd = (
                            stats_map.get("max_drawdown")
                            if stats_map.get("max_drawdown") is not None
                            else fallback_stats.get("Max DD")
                        )

                        oos_summary.append(
                            {
                                "Fund": col,
                                "Total Return": _fmt_pct(total_ret, 1),
                                "Ann. Vol": _fmt_pct(vol, 1),
                                "Months": len(returns),
                                "CAGR": _fmt_pct(cagr, 1),
                                "Sharpe": _fmt_ratio(sharpe),
                                "Sortino": _fmt_ratio(sortino),
                                "Info Ratio": _fmt_ratio(info_ratio),
                                "Max DD": _fmt_pct(max_dd, 1),
                            }
                        )

                if oos_summary:
                    st.dataframe(
                        pd.DataFrame(oos_summary),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Show raw returns in expander
                with st.expander("View monthly returns detail"):
                    display_oos = out_df[cols_to_show].copy()
                    display_oos.index = pd.to_datetime(display_oos.index).strftime(
                        "%Y-%m"
                    )
                    for col in display_oos.columns:
                        display_oos[col] = display_oos[col].apply(
                            lambda x: _fmt_pct(x, 2) if pd.notna(x) else "—"
                        )
                    st.dataframe(display_oos, use_container_width=True)
        else:
            st.caption("No out-of-sample returns available.")

    with tabs[3]:
        # Period portfolio return
        out_user = period_data["out_user_stats"]
        out_ew = period_data["out_ew_stats"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Weighted Portfolio (This Period)**")
            if out_user:
                stats = out_user if isinstance(out_user, dict) else vars(out_user)
                metrics_to_show = ["cagr", "vol", "sharpe", "sortino", "max_dd"]
                for m in metrics_to_show:
                    val = stats.get(m) or stats.get(m.replace("_", ""))
                    if val is not None:
                        if m in ["cagr", "vol", "max_dd"]:
                            st.metric(m.upper().replace("_", " "), _fmt_pct(val, 1))
                        else:
                            st.metric(m.title().replace("_", " "), _fmt_ratio(val))
            else:
                st.caption("No weighted portfolio stats available.")

        with col2:
            st.markdown("**Equal-Weight Portfolio (This Period)**")
            if out_ew:
                stats = out_ew if isinstance(out_ew, dict) else vars(out_ew)
                metrics_to_show = ["cagr", "vol", "sharpe", "sortino", "max_dd"]
                for m in metrics_to_show:
                    val = stats.get(m) or stats.get(m.replace("_", ""))
                    if val is not None:
                        if m in ["cagr", "vol", "max_dd"]:
                            st.metric(m.upper().replace("_", " "), _fmt_pct(val, 1))
                        else:
                            st.metric(m.title().replace("_", " "), _fmt_ratio(val))
            else:
                st.caption("No equal-weight portfolio stats available.")


def _render_period_breakdown(result) -> None:
    """Render period-by-period breakdown with full reproducibility data."""
    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", [])

    if not period_results:
        st.caption("No period-by-period data available.")
        return

    # Build period data
    periods_data = [
        _build_period_detail(res, i + 1) for i, res in enumerate(period_results)
    ]

    # Show weights across all rebalance dates (sub-period visibility)
    try:
        weights_df = _build_period_weights_df(result)
        if not weights_df.empty:
            with st.expander("View portfolio weights by rebalance date"):
                pivot = weights_df.pivot_table(
                    index="Period Start",
                    columns="Fund",
                    values="Weight",
                    aggfunc="first",
                    fill_value=0.0,
                ).sort_index()
                pivot_fmt = pivot.applymap(lambda x: _fmt_pct(float(x), 1))
                st.dataframe(pivot_fmt, use_container_width=True)
    except Exception:
        # Defensive: weights visibility should not break the results page.
        pass

    # Period selector
    period_options = [
        f"Period {p['period_num']}: {p['out_start']} to {p['out_end']}"
        for p in periods_data
    ]

    selected_period = st.selectbox(
        "Select period to examine:",
        options=range(len(period_options)),
        format_func=lambda x: period_options[x],
    )

    if selected_period is not None:
        _render_single_period(periods_data[selected_period])


# =============================================================================
# Portfolio by Period (Download) - Fixed to only include portfolio funds
# =============================================================================


def _build_period_weights_df(result) -> pd.DataFrame:
    """Build DataFrame of weights by period for download - only portfolio funds."""
    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", [])

    if not period_results:
        return pd.DataFrame()

    portfolio_by_period = _get_portfolio_funds_by_period(result)

    rows = []
    for res in period_results:
        period = res.get("period", ("", "", "", ""))
        out_start = period[2] if len(period) > 2 else ""
        rebalance_weights = res.get("rebalance_weights")
        ew_weights = res.get("ew_weights", {})
        fund_weights = res.get("fund_weights", {})

        # Prefer intra-period rebalance weights when available.
        if isinstance(rebalance_weights, pd.DataFrame) and not rebalance_weights.empty:
            for reb_date, w_row in rebalance_weights.iterrows():
                stamp = pd.to_datetime(reb_date).date().isoformat()
                for fund, weight in w_row.items():
                    try:
                        w = float(weight)
                    except Exception:
                        continue
                    if abs(w) <= 1e-12:
                        continue
                    rows.append(
                        {
                            "Period Start": stamp,
                            "Period End": period[3] if len(period) > 3 else "",
                            "Fund": str(fund),
                            "Weight": w,
                        }
                    )
            continue

        # Get funds actually in portfolio this period
        portfolio_funds = portfolio_by_period.get(out_start, set())
        weights = fund_weights or ew_weights or {}

        for fund in portfolio_funds:
            weight = weights.get(fund, 0.0)
            rows.append(
                {
                    "Period Start": out_start,
                    "Period End": period[3] if len(period) > 3 else "",
                    "Fund": fund,
                    "Weight": weight,
                }
            )

    df = pd.DataFrame(rows)
    return df


def _build_period_returns_df(result) -> pd.DataFrame:
    """Build DataFrame of returns by period for download - only portfolio funds."""
    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", [])

    if not period_results:
        return pd.DataFrame()

    portfolio_by_period = _get_portfolio_funds_by_period(result)

    all_dfs = []
    for res in period_results:
        period = res.get("period", ("", "", "", ""))
        out_start = period[2] if len(period) > 2 else ""
        out_df = res.get("out_sample_scaled")

        # Get funds actually in portfolio this period
        portfolio_funds = portfolio_by_period.get(out_start, set())

        if isinstance(out_df, pd.DataFrame) and not out_df.empty:
            # Filter to only portfolio funds
            cols = [c for c in out_df.columns if c in portfolio_funds]
            if cols:
                oos = out_df[cols].copy()
                oos["Period Start"] = out_start
                oos["Period End"] = period[3] if len(period) > 3 else ""
                all_dfs.append(oos.reset_index())

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def _render_download_section(result) -> None:
    """Render download buttons for portfolio data."""
    weights_df = _build_period_weights_df(result)
    returns_df = _build_period_returns_df(result)
    changes_df = _extract_manager_changes(result)
    summary_df, risk_df = _compute_fund_holding_periods(result)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not weights_df.empty:
            csv = weights_df.to_csv(index=False)
            st.download_button(
                "📊 Weights",
                csv,
                "portfolio_weights.csv",
                "text/csv",
            )
        else:
            st.caption("No weights")

    with col2:
        if not returns_df.empty:
            csv = returns_df.to_csv(index=False)
            st.download_button(
                "📈 Returns",
                csv,
                "portfolio_returns.csv",
                "text/csv",
            )
        else:
            st.caption("No returns")

    with col3:
        if not changes_df.empty:
            csv = changes_df.to_csv(index=False)
            st.download_button(
                "🔄 Changes",
                csv,
                "manager_changes.csv",
                "text/csv",
            )
        else:
            st.caption("No changes")

    with col4:
        if not summary_df.empty:
            # Combine summary and risk stats
            if not risk_df.empty:
                combined = summary_df.merge(risk_df, on="Manager", how="left")
            else:
                combined = summary_df
            csv = combined.to_csv(index=False)
            st.download_button(
                "📋 Holdings",
                csv,
                "fund_holdings.csv",
                "text/csv",
            )
        else:
            st.caption("No holdings")

    st.divider()
    st.subheader("📎 Excel Workbook")
    st.caption(
        "Downloads a Phase-1 style workbook including summary + per-period sheets, "
        "with out-of-sample return/risk metrics and the requested number formats."
    )

    def _build_xlsx_bytes() -> bytes:
        import tempfile

        from trend_analysis.export import export_phase1_workbook

        details = getattr(result, "details", {}) or {}
        period_results = details.get("period_results", [])
        if not isinstance(period_results, list) or not period_results:
            return b""

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_phase1_workbook(period_results, tmp_path)
            return Path(tmp_path).read_bytes()
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    xlsx_bytes = _build_xlsx_bytes()
    if xlsx_bytes:
        st.download_button(
            "⬇️ Download Phase-1 Workbook (XLSX)",
            data=xlsx_bytes,
            file_name="trend_phase1_workbook.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.caption("Run a multi-period analysis to enable workbook export.")


# =============================================================================
# Current Exposures - Fixed to get from last period
# =============================================================================


def _get_current_exposures(result) -> pd.Series | None:
    """Get current portfolio exposures from the last period."""
    details = getattr(result, "details", {}) or {}
    period_results = details.get("period_results", [])

    if not period_results:
        # Try other sources
        analysis = getattr(result, "analysis", None)
        if analysis is not None:
            return analysis.exposures
        risk_diag = details.get("risk_diagnostics")
        if isinstance(risk_diag, dict):
            return risk_diag.get("final_weights")
        return None

    # Get weights from last period
    last_period = period_results[-1]
    fund_weights = last_period.get("fund_weights") or last_period.get("ew_weights")

    if fund_weights:
        return pd.Series(fund_weights, name="Weight").sort_values(ascending=False)

    return None


# =============================================================================
# Charts
# =============================================================================


def _render_charts(result) -> None:
    """Render portfolio charts."""
    analysis = getattr(result, "analysis", None)
    details = result.details if hasattr(result, "details") else {}
    returns = None

    if analysis is not None:
        returns = analysis.returns
    if returns is None:
        returns = getattr(result, "portfolio", None)
    if returns is None:
        returns = details.get("portfolio_equal_weight_combined")

    if not isinstance(returns, pd.Series) or returns.empty:
        st.info("Run the analysis to see portfolio charts.")
        return

    equity = _prepare_equity_series(returns)
    drawdown = _prepare_drawdown(equity)
    rolling = _rolling_sharpe(returns)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cumulative Portfolio Returns")
        st.altair_chart(charts.equity_chart(equity), use_container_width=True)
    with c2:
        st.subheader("Drawdown")
        st.altair_chart(charts.drawdown_chart(drawdown), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Rolling Sharpe (12m)")
        st.altair_chart(charts.rolling_sharpe_chart(rolling), use_container_width=True)
    with c4:
        turnover = getattr(result, "turnover", None)
        if turnover is None and analysis is not None:
            turnover = analysis.turnover
        if turnover is None:
            diag_obj = details.get("risk_diagnostics")
            if isinstance(diag_obj, dict):
                turnover = diag_obj.get("turnover")
        st.subheader("Turnover")
        if isinstance(turnover, pd.Series) and not turnover.empty:
            st.altair_chart(charts.turnover_chart(turnover), use_container_width=True)
        else:
            st.caption("Turnover data unavailable.")

    # Current Exposures - use fixed function
    exposures = _get_current_exposures(result)
    st.subheader("Current Exposures")
    if exposures is not None and not exposures.empty:
        st.altair_chart(charts.exposure_chart(exposures), use_container_width=True)
    else:
        st.caption("Exposure breakdown unavailable.")


# =============================================================================
# Summary Metrics
# =============================================================================


def _render_summary(result, returns: pd.Series | None) -> None:
    """Render summary metrics section."""
    metrics = result.metrics

    st.subheader("📊 Portfolio Performance Summary")
    if returns is not None and isinstance(returns, pd.Series) and not returns.empty:
        _render_trailing_stats(returns)
    elif metrics is not None and not metrics.empty:
        st.dataframe(metrics)
    else:
        st.caption("No summary metrics available.")


# =============================================================================
# Main Render Function
# =============================================================================


def render_results_page() -> None:
    """Main render function for the Results page."""
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

    # Apply fund selection (if present) so analysis uses the intended subset.
    applied_funds = st.session_state.get("analysis_fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = st.session_state.get("fund_columns")
    if not isinstance(applied_funds, list):
        applied_funds = []

    selected_rf = st.session_state.get("selected_risk_free")
    info_ratio_benchmark = model_state.get("info_ratio_benchmark")
    prohibited = {selected_rf, benchmark, info_ratio_benchmark} - {None}

    # Policy: benchmark/index columns (including Info Ratio benchmark) and RF
    # are never investable funds.
    sanitized_funds = [
        c for c in applied_funds if c in df.columns and c not in prohibited
    ]
    removed = [c for c in applied_funds if c in df.columns and c in prohibited]
    keep_cols = list(sanitized_funds)
    for extra in (selected_rf, benchmark):
        if extra and extra in df.columns and extra not in keep_cols:
            keep_cols.append(extra)

    if keep_cols:
        df_for_analysis = df[keep_cols]
        st.caption(f"Using {len(sanitized_funds)} selected funds for analysis")
        if removed:
            st.warning(
                "Removed non-investable columns from fund selection: "
                + ", ".join(sorted(set(removed)))
            )
    else:
        df_for_analysis = df
        st.caption("Using all columns for analysis")

    with st.expander("Debug: download current parameters", expanded=False):
        params = {
            "uploaded_filename": st.session_state.get("uploaded_filename"),
            "data_loaded_key": st.session_state.get("data_loaded_key"),
            "data_fingerprint": st.session_state.get("data_fingerprint"),
            "selected_benchmark": benchmark,
            "selected_risk_free": selected_rf,
            "analysis_fund_columns": st.session_state.get("analysis_fund_columns"),
            "fund_columns": st.session_state.get("fund_columns"),
            "model_state": model_state,
        }

        payload = json.dumps(params, indent=2, sort_keys=True, default=str).encode(
            "utf-8"
        )
        st.download_button(
            "Download parameters (JSON)",
            data=payload,
            file_name="trend_run_parameters.json",
            mime="application/json",
        )

        def _apply_params(uploaded_params: dict[str, Any]) -> None:
            for k in (
                "selected_benchmark",
                "selected_risk_free",
                "analysis_fund_columns",
                "fund_columns",
                "model_state",
                "uploaded_filename",
                "data_loaded_key",
                "data_fingerprint",
            ):
                if k in uploaded_params:
                    st.session_state[k] = uploaded_params.get(k)
            analysis_runner.clear_cached_analysis()

        st.markdown("**Import parameters (JSON)**")
        st.caption(
            "Codespaces tip: the file picker can’t browse the Codespace filesystem. "
            "Use either paste, or load from a workspace path (e.g. tmp/trend_run_parameters.json)."
        )

        import_mode = st.radio(
            "Import method",
            options=["Paste JSON", "Load from workspace path", "Upload file"],
            horizontal=True,
            key="run_params_import_mode",
        )

        uploaded_params: dict[str, Any] | None = None
        if import_mode == "Paste JSON":
            pasted = st.text_area(
                "Paste JSON here",
                value="",
                height=180,
                key="run_params_paste",
            )
            if pasted.strip():
                try:
                    obj = json.loads(pasted)
                    uploaded_params = obj if isinstance(obj, dict) else None
                    if uploaded_params is None:
                        st.error("JSON must be an object (top-level dictionary).")
                except Exception as exc:
                    st.error(f"Could not parse JSON: {exc}")

        elif import_mode == "Load from workspace path":
            default_path = "tmp/trend_run_parameters.json"
            rel_path = st.text_input(
                "Workspace-relative path",
                value=default_path,
                key="run_params_workspace_path",
                help="Example: tmp/trend_run_parameters.json",
            )
            if st.button("Load JSON from path", key="btn_load_params_from_path"):
                try:
                    repo_root = Path.cwd()
                    candidate = (repo_root / rel_path).resolve()
                    if not str(candidate).startswith(str(repo_root.resolve())):
                        raise ValueError("Path must be inside the workspace")
                    text = candidate.read_text(encoding="utf-8")
                    obj = json.loads(text)
                    uploaded_params = obj if isinstance(obj, dict) else None
                    if uploaded_params is None:
                        st.error("JSON must be an object (top-level dictionary).")
                except Exception as exc:
                    st.error(f"Could not load JSON: {exc}")

        else:  # Upload file
            uploaded = st.file_uploader(
                "Upload trend_run_parameters.json",
                type=["json"],
                accept_multiple_files=False,
                key="uploaded_run_parameters_json",
            )
            if uploaded is not None:
                try:
                    uploaded_text = uploaded.getvalue().decode("utf-8")
                    obj = json.loads(uploaded_text)
                    uploaded_params = obj if isinstance(obj, dict) else None
                    if uploaded_params is None:
                        st.error("JSON must be an object (top-level dictionary).")
                except Exception as exc:
                    st.error(f"Could not parse JSON: {exc}")

        if isinstance(uploaded_params, dict):
            st.success("Parameters loaded.")
            with st.expander("Preview imported parameters", expanded=False):
                st.json(uploaded_params)

            apply_cols = st.columns([1, 3])
            with apply_cols[0]:
                if st.button(
                    "Apply imported parameters", key="btn_apply_imported_params"
                ):
                    _apply_params(uploaded_params)
                    st.success("Applied. Re-run analysis to reproduce the run.")
                    st.rerun()
            with apply_cols[1]:
                st.caption(
                    "Applying will overwrite your current selections and model settings for this session."
                )
    run_key = _current_run_key(model_state, benchmark)
    cached_key = st.session_state.get("analysis_result_key")
    result = st.session_state.get("analysis_result") if cached_key == run_key else None

    st.markdown("Run the analysis to generate performance and risk diagnostics.")
    run_clicked = st.button("Run analysis", type="primary")

    if run_clicked or result is None:
        with st.spinner("Running analysis…"):
            try:
                base_hash = st.session_state.get("data_fingerprint")
                cols_hash = hashlib.sha256(
                    json.dumps(list(df_for_analysis.columns), default=str).encode(
                        "utf-8"
                    )
                ).hexdigest()[:12]
                data_hash = f"{base_hash}:{cols_hash}" if base_hash else cols_hash
                # Pass risk-free column explicitly into the config used by the
                # backend (without mutating the session model_state).
                effective_model_state = dict(model_state)
                effective_model_state["risk_free_column"] = selected_rf

                result = analysis_runner.run_analysis(
                    df_for_analysis,
                    effective_model_state,
                    benchmark,
                    data_hash=data_hash,
                )
            except Exception as exc:
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
        st.info("Click Run Analysis to generate a report.")
        return

    summary, detail = _diagnostic_message(result)
    if summary:
        st.error(summary)
        if detail:
            st.caption(detail)
        return

    fallback = getattr(result, "fallback_info", None)
    if isinstance(fallback, dict):
        st.warning(
            "Weight engine fallback: using equal weights because "
            f"{fallback.get('engine', 'unknown engine')} failed."
        )

    # Get returns for stats
    analysis = getattr(result, "analysis", None)
    details = getattr(result, "details", {}) or {}
    returns = None
    if analysis is not None:
        returns = analysis.returns
    if returns is None:
        returns = getattr(result, "portfolio", None)
    if returns is None:
        returns = details.get("portfolio_equal_weight_combined")
    if (
        returns is None
        or not isinstance(returns, pd.Series)
        or (isinstance(returns, pd.Series) and returns.empty)
    ):
        if isinstance(details, dict):
            returns = _portfolio_series_from_details(details)

    # Multi-period info
    period_count = getattr(result, "period_count", 0) or details.get("period_count", 0)
    sim_start, sim_end = _get_simulation_date_range(result)

    # ==========================================================================
    # REORGANIZED LAYOUT FOR REPRODUCIBILITY
    # ==========================================================================

    # Header with simulation info
    if period_count > 0:
        st.success(
            f"✅ Simulation complete: **{period_count} periods** from {sim_start} to {sim_end}"
        )

    # Create main tabs for organized navigation
    main_tabs = st.tabs(
        [
            "📊 Summary",
            "🔬 Period Analysis",
            "📈 Visualizations",
            "📋 Fund Details",
            "💾 Export",
        ]
    )

    # ==========================================================================
    # TAB 1: SUMMARY - Total portfolio performance
    # ==========================================================================
    with main_tabs[0]:
        st.header("Total Portfolio Performance")
        st.caption("Aggregated statistics computed from all out-of-sample returns")

        # Selection criteria used
        st.subheader("🎯 Selection Criteria Used")
        with st.expander("View selection parameters", expanded=True):
            _render_selection_criteria(result)

        st.divider()

        # Overall performance metrics
        st.subheader("📊 Cumulative Out-of-Sample Performance")
        if returns is not None and isinstance(returns, pd.Series) and not returns.empty:
            _render_trailing_stats(returns)

            # Key metrics in prominent display
            if len(returns) >= 12:
                total_ret = (1 + returns).prod() - 1
                years = len(returns) / 12
                cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
                vol = returns.std() * np.sqrt(12)
                sharpe = cagr / vol if vol > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", _fmt_pct(total_ret, 1))
                col2.metric("CAGR", _fmt_pct(cagr, 1))
                col3.metric("Ann. Volatility", _fmt_pct(vol, 1))
                col4.metric("Sharpe Ratio", _fmt_ratio(sharpe))
        else:
            st.caption("Run analysis to see performance metrics.")

    # ==========================================================================
    # TAB 2: PERIOD ANALYSIS - Period-by-period reproducibility
    # ==========================================================================
    with main_tabs[1]:
        st.header("Period-by-Period Analysis")
        st.caption(
            "Examine the data chain: In-sample metrics → Selection decisions → "
            "Out-of-sample returns → Period performance"
        )

        if period_count > 0:
            _render_period_breakdown(result)
        else:
            st.info("Single-period analysis does not have period breakdown.")
            # Show single period data if available
            if details:
                score_frame = details.get("score_frame")
                if isinstance(score_frame, pd.DataFrame) and not score_frame.empty:
                    st.subheader("In-Sample Metrics (All Candidates)")
                    st.dataframe(score_frame, use_container_width=True)

    # ==========================================================================
    # TAB 3: VISUALIZATIONS - Charts
    # ==========================================================================
    with main_tabs[2]:
        st.header("Portfolio Visualizations")
        _render_charts(result)

    # ==========================================================================
    # TAB 4: FUND DETAILS - Holdings and changes
    # ==========================================================================
    with main_tabs[3]:
        st.header("Fund Details")

        # Manager changes
        st.subheader("🔄 Manager Hiring & Termination Timeline")
        _render_manager_changes(result)

        st.divider()

        # Fund holding periods with risk stats
        st.subheader("📋 Fund Holdings & Out-of-Sample Performance")
        _render_fund_holdings(result)

    # ==========================================================================
    # TAB 5: EXPORT - Download data
    # ==========================================================================
    with main_tabs[4]:
        st.header("Export Data")
        st.caption("Download detailed data for external analysis and verification")

        _render_download_section(result)

        # Additional: Full period data export
        st.divider()
        st.subheader("📄 Complete Period Data")

        if period_count > 0:
            period_results = details.get("period_results", [])

            # Build comprehensive export
            all_period_data = []
            for i, res in enumerate(period_results):
                period = res.get("period", ("", "", "", ""))
                score_frame = res.get("score_frame")
                selected = res.get("selected_funds", [])
                weights = res.get("fund_weights", {})

                if isinstance(score_frame, pd.DataFrame) and not score_frame.empty:
                    for fund in score_frame.index:
                        weight_value = float(weights.get(fund, 0.0) or 0.0)
                        row = {
                            "Period": i + 1,
                            "In-Sample Start": period[0] if len(period) > 0 else "",
                            "In-Sample End": period[1] if len(period) > 1 else "",
                            "Out-Sample Start": period[2] if len(period) > 2 else "",
                            "Out-Sample End": period[3] if len(period) > 3 else "",
                            "Fund": fund,
                            "Selected": "Yes" if fund in selected else "No",
                            # Keep numeric weights for exports; format at display time.
                            "Weight": weight_value,
                        }
                        # Add all score columns
                        for col in score_frame.columns:
                            row[f"InSample_{col}"] = score_frame.loc[fund, col]
                        all_period_data.append(row)

            if all_period_data:
                full_df = pd.DataFrame(all_period_data)
                base_cols = [
                    "Period",
                    "In-Sample Start",
                    "In-Sample End",
                    "Out-Sample Start",
                    "Out-Sample End",
                    "Fund",
                    "Selected",
                    "Weight",
                ]
                metric_cols = [c for c in full_df.columns if c.startswith("InSample_")]
                ordered_cols = [
                    c for c in base_cols if c in full_df.columns
                ] + metric_cols
                full_df = full_df.loc[:, ordered_cols]
                csv = full_df.to_csv(index=False)
                st.download_button(
                    "📊 Download Complete Period Analysis (CSV)",
                    csv,
                    "complete_period_analysis.csv",
                    "text/csv",
                )
                st.caption(
                    "Contains in-sample metrics for all candidates in all periods, "
                    "with selection status and weights."
                )


if _should_auto_render():
    render_results_page()
