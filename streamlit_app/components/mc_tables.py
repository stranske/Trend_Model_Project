"""Monte Carlo summary table helpers for the Streamlit app."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from trend_analysis.monte_carlo.aggregator import aggregate_monte_carlo_results

DEFAULT_QUANTILES: tuple[float, float] = (0.05, 0.5)

_SHARPE_ALIASES = ("sharpe", "sharp", "sharperatio")
_MAX_DD_ALIASES = ("maxdrawdown", "maxdd", "drawdown", "max_drawdown")
_TERMINAL_ALIASES = (
    "terminalwealth",
    "terminalvalue",
    "finalwealth",
    "endingwealth",
    "terminal_wealth",
    "nav",
)


def _canonical_metric(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _resolve_metric(metrics: Iterable[str], aliases: Sequence[str]) -> str | None:
    alias_set = {_canonical_metric(alias) for alias in aliases}
    for metric in metrics:
        if _canonical_metric(metric) in alias_set:
            return metric
    return None


def _extract_results_frame(results: Any) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results
    if hasattr(results, "results_frame"):
        frame = getattr(results, "results_frame")
        if isinstance(frame, pd.DataFrame):
            return frame
    if isinstance(results, Mapping):
        for key in ("results_frame", "results", "frame"):
            frame = results.get(key)
            if isinstance(frame, pd.DataFrame):
                return frame
    return pd.DataFrame()


def build_summary_table(
    results: Any,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Build a summary table with median and lower-tail metrics per strategy."""

    columns = [
        "Strategy",
        "Sharpe (median)",
        "Sharpe (5th)",
        "Max DD (median)",
        "Max DD (5th)",
        "Terminal Wealth",
    ]
    results_frame = _extract_results_frame(results)
    if results_frame.empty:
        return pd.DataFrame(columns=columns)

    aggregation = aggregate_monte_carlo_results(results_frame, quantiles=list(quantiles))
    quantiles_frame = aggregation.quantiles_frame
    if quantiles_frame.empty:
        return pd.DataFrame(columns=columns)

    frame = quantiles_frame.copy()
    for col in ("strategy", "metric", "quantile", "value"):
        if col not in frame.columns:
            frame[col] = pd.NA
    frame["quantile"] = pd.to_numeric(frame["quantile"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["strategy", "metric", "quantile"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    metric_names = [str(metric) for metric in frame["metric"].dropna().unique()]
    sharpe_metric = _resolve_metric(metric_names, _SHARPE_ALIASES)
    max_dd_metric = _resolve_metric(metric_names, _MAX_DD_ALIASES)
    terminal_metric = _resolve_metric(metric_names, _TERMINAL_ALIASES)

    grouped = (
        frame.groupby(["strategy", "metric", "quantile"], dropna=False)["value"]
        .mean()
        .reset_index()
    )

    quantile_set = sorted(set(float(q) for q in quantiles))
    q05 = 0.05 if 0.05 in quantile_set else (quantile_set[0] if quantile_set else None)
    q50 = 0.5 if 0.5 in quantile_set else (quantile_set[-1] if quantile_set else None)

    def _lookup(metric: str | None, quantile: float | None) -> dict[str, float | None]:
        if metric is None or quantile is None:
            return {}
        subset = grouped[(grouped["metric"] == metric) & np.isclose(grouped["quantile"], quantile)]
        return dict(zip(subset["strategy"], subset["value"]))

    sharpe_median = _lookup(sharpe_metric, q50)
    sharpe_5th = _lookup(sharpe_metric, q05)
    max_dd_median = _lookup(max_dd_metric, q50)
    max_dd_5th = _lookup(max_dd_metric, q05)
    terminal_median = _lookup(terminal_metric, q50)

    strategies = sorted(set(grouped["strategy"].dropna().astype(str)))
    rows = []
    for strategy in strategies:
        rows.append(
            {
                "Strategy": strategy,
                "Sharpe (median)": sharpe_median.get(strategy),
                "Sharpe (5th)": sharpe_5th.get(strategy),
                "Max DD (median)": max_dd_median.get(strategy),
                "Max DD (5th)": max_dd_5th.get(strategy),
                "Terminal Wealth": terminal_median.get(strategy),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def render_summary_table(
    results: Any,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Render the summary table to Streamlit and return the frame."""

    table = build_summary_table(results, quantiles=quantiles)
    st.dataframe(table, use_container_width=True, hide_index=True)
    return table
