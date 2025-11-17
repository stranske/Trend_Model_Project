"""Summary metrics for simulation results."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from . import annual_return, information_ratio, max_drawdown, sharpe_ratio, volatility
from .turnover import realized_turnover, turnover_cost
from trend_analysis.costs import CostModel


def summary_table(
    returns: pd.Series,
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
    benchmark: pd.Series | float | None = None,
    periods_per_year: int = 12,
    transaction_cost_bps: float = 0.0,
    cost_model: CostModel | None = None,
) -> pd.DataFrame:
    """Return a one-column DataFrame of core performance statistics.

    Parameters
    ----------
    returns:
        Net periodic returns of the strategy.
    weights:
        History of portfolio weights by period.
    benchmark:
        Optional benchmark series or scalar.
    periods_per_year:
        Frequency of ``returns``.
    transaction_cost_bps:
        Legacy basis-point cost rate per unit turnover.
    cost_model:
        Optional :class:`~trend_analysis.costs.CostModel` overriding
        ``transaction_cost_bps`` for more granular control.
    """

    bench = 0.0 if benchmark is None else benchmark

    model = cost_model or CostModel.from_legacy(transaction_cost_bps)
    turn_df = realized_turnover(weights)
    cost_series = turnover_cost(weights, model)
    net_returns = returns - cost_series.reindex(returns.index).fillna(0.0)

    cagr = annual_return(net_returns, periods_per_year=periods_per_year)
    vol = volatility(net_returns, periods_per_year=periods_per_year)
    mdd = max_drawdown(net_returns)
    ir = information_ratio(net_returns, bench, periods_per_year=periods_per_year)
    sharpe = sharpe_ratio(net_returns, risk_free=0.0, periods_per_year=periods_per_year)
    turn = float(turn_df["turnover"].mean())
    cost = float(cost_series.mean())
    hit_rate = float((net_returns > 0).mean())

    data = {
        "CAGR": float(cagr),
        "vol": float(vol),
        "max_drawdown": float(mdd),
        "information_ratio": float(ir),
        "sharpe": float(sharpe),
        "turnover": float(turn),
        "cost_impact": float(cost),
        "hit_rate": float(hit_rate),
    }
    return pd.DataFrame.from_dict(data, orient="index", columns=["value"])


__all__ = ["summary_table"]
