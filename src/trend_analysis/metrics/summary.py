"""Summary metrics for simulation results."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from . import (
    annual_return,
    volatility,
    max_drawdown,
    information_ratio,
    sharpe_ratio,
)
from ..viz.charts import turnover_series


def summary_table(
    returns: pd.Series,
    weights: Mapping[pd.Timestamp, pd.Series] | pd.DataFrame,
    benchmark: pd.Series | float | None = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Return a one-column DataFrame of core performance statistics."""

    bench = 0.0 if benchmark is None else benchmark

    cagr = annual_return(returns, periods_per_year=periods_per_year)
    vol = volatility(returns, periods_per_year=periods_per_year)
    mdd = max_drawdown(returns)
    ir = information_ratio(returns, bench, periods_per_year=periods_per_year)
    sharpe = sharpe_ratio(returns, risk_free=0.0, periods_per_year=periods_per_year)
    turn = float(turnover_series(weights).mean())
    hit_rate = float((returns > 0).mean())

    data = {
        "CAGR": float(cagr),
        "vol": float(vol),
        "max_drawdown": float(mdd),
        "information_ratio": float(ir),
        "sharpe": float(sharpe),
        "turnover": float(turn),
        "hit_rate": float(hit_rate),
    }
    return pd.DataFrame.from_dict(data, orient="index", columns=["value"])


__all__ = ["summary_table"]

