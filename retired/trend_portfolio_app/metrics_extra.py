from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12


def total_return(r: pd.Series) -> float:
    rf = r.astype(float).fillna(0.0)
    prod_val = float(np.prod(1.0 + rf.values.astype(float)).item())
    return float(prod_val - 1.0)


def annualized_return(r: pd.Series, idx: pd.DatetimeIndex) -> float:
    n_years = max((idx[-1] - idx[0]).days / 365.25, EPS)
    tr = float(total_return(r))
    return float((1.0 + tr) ** (1.0 / float(n_years)) - 1.0)


def volatility(r: pd.Series, idx: pd.DatetimeIndex) -> float:
    return float(r.astype(float).std(ddof=0) * np.sqrt(12))


def sharpe(r: pd.Series, idx: pd.DatetimeIndex, rf_annual: float = 0.0) -> float:
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    ex = r.astype(float) - float(rf_monthly)
    vol = ex.std(ddof=0) * np.sqrt(12)
    if vol < EPS:
        return 0.0
    return float(ex.mean() * 12 / vol)


def sortino(r: pd.Series, idx: pd.DatetimeIndex, rf_annual: float = 0.0) -> float:
    rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
    ex = r.astype(float) - float(rf_monthly)
    downside = ex.copy()
    downside[downside > 0] = 0
    dd = np.sqrt((downside**2).mean()) * np.sqrt(12)
    if dd < EPS:
        return 0.0
    return float(ex.mean() * 12 / dd)


def max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    return float(dd.min())


def drawdown_duration(cum: pd.Series) -> int:
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    duration = 0
    max_duration = 0
    for x in dd:
        if x < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return int(max_duration)


def ulcer_index(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = (cum / roll_max - 1.0) * 100
    return float(np.sqrt((dd**2).mean()))


def hit_rate(r: pd.Series) -> float:
    return float((r > 0).mean())


AVAILABLE_METRICS: dict[str, dict[str, Any]] = {
    "return_ann": {"fn": annualized_return, "higher_is_better": True},
    "vol": {"fn": volatility, "higher_is_better": False},
    "sharpe": {"fn": sharpe, "higher_is_better": True},
    "sortino": {"fn": sortino, "higher_is_better": True},
    "drawdown": {
        "fn": lambda r, idx: float(max_drawdown((1 + r).cumprod())),
        "higher_is_better": False,
    },
    "dd_duration": {
        "fn": lambda r, idx: int(drawdown_duration((1 + r).cumprod())),
        "higher_is_better": False,
    },
    "ulcer": {
        "fn": lambda r, idx: float(ulcer_index((1 + r).cumprod())),
        "higher_is_better": False,
    },
    "hit_rate": {
        "fn": lambda r, idx: float(hit_rate(r)),
        "higher_is_better": True,
    },
}
