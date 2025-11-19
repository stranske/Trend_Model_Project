"""Backtesting harness with walk-forward support and transaction costs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from backtest import shift_by_execution_lag

WindowMode = Literal["rolling", "expanding"]


@dataclass(frozen=True)
class CostModel:
    """Linear transaction cost model applied to turnover events."""

    bps_per_trade: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value in {
            "bps_per_trade": self.bps_per_trade,
            "slippage_bps": self.slippage_bps,
        }.items():
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

    def apply(self, turnover: float) -> float:
        if turnover <= 0:
            return 0.0
        multiplier = (self.bps_per_trade + self.slippage_bps) / 10000.0
        return float(turnover) * multiplier

    def as_dict(self) -> Dict[str, float]:
        return {
            "bps_per_trade": float(self.bps_per_trade),
            "slippage_bps": float(self.slippage_bps),
        }


@dataclass
class BacktestResult:
    """Container for backtest artifacts and computed performance statistics."""

    returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    per_period_turnover: pd.Series
    transaction_costs: pd.Series
    rolling_sharpe: pd.Series
    drawdown: pd.Series
    metrics: Dict[str, float]
    cost_model: CostModel
    calendar: pd.DatetimeIndex
    window_mode: WindowMode
    window_size: int
    training_windows: Mapping[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]]
    execution_lag: int = 1

    def summary(self) -> Dict[str, object]:
        """Return a JSON-serializable summary of the backtest metrics."""

        return {
            "window_mode": self.window_mode,
            "window_size": self.window_size,
            "execution_lag": self.execution_lag,
            "calendar": [ts.isoformat() for ts in self.calendar],
            "metrics": {k: _to_float(v) for k, v in self.metrics.items()},
            "returns": _series_to_dict(self.returns),
            "rolling_sharpe": _series_to_dict(self.rolling_sharpe),
            "drawdown": _series_to_dict(self.drawdown),
            "equity_curve": _series_to_dict(self.equity_curve),
            "turnover": _series_to_dict(self.turnover),
            "per_period_turnover": _series_to_dict(self.per_period_turnover),
            "transaction_costs": _series_to_dict(self.transaction_costs),
            "weights": _weights_to_dict(self.weights),
            "training_windows": {
                ts.isoformat(): {
                    "start": window[0].isoformat(),
                    "end": window[1].isoformat(),
                }
                for ts, window in self.training_windows.items()
            },
            "cost_model": self.cost_model.as_dict(),
        }

    def to_json(self, **dumps_kwargs: Any) -> str:
        """Serialise :meth:`summary` to JSON for downstream consumers."""

        return json.dumps(self.summary(), default=_json_default, **dumps_kwargs)


def run_backtest(
    returns: pd.DataFrame,
    strategy: Callable[[pd.DataFrame], pd.Series | Mapping[str, float]],
    *,
    rebalance_freq: str,
    window_size: int,
    window_mode: WindowMode = "rolling",
    transaction_cost_bps: float = 0.0,
    min_trade: float,
    rolling_sharpe_window: int | None = None,
    initial_weights: Mapping[str, float] | None = None,
    cost_model: CostModel | None = None,
    execution_lag: int = 1,
) -> BacktestResult:
    """Run a walk-forward backtest with a fixed rebalance calendar.

    Args:
        min_trade: Minimum total absolute weight change required to execute a
            rebalance. Smaller proposals are ignored to suppress micro-churn.
        execution_lag: Number of periods between generating a signal and being
            allowed to apply the resulting weights. ``1`` enforces the standard
            "compute on close, trade next bar" convention and prevents
            same-day look-ahead.
    """

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_mode not in {"rolling", "expanding"}:
        raise ValueError("window_mode must be 'rolling' or 'expanding'")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if min_trade < 0:
        raise ValueError("min_trade must be non-negative")
    if execution_lag <= 0:
        raise ValueError("execution_lag must be a positive integer to avoid look-ahead")

    model = cost_model or CostModel(bps_per_trade=transaction_cost_bps)

    data = _prepare_returns(returns)
    if data.empty:
        raise ValueError("returns must contain at least one row")

    calendar = _rebalance_calendar(data.index, rebalance_freq)
    if not len(calendar):
        raise ValueError("rebalance calendar produced no dates – check frequency")

    periods_per_year = _infer_periods_per_year(data.index)
    roll_window = rolling_sharpe_window or min(
        window_size, max(1, periods_per_year // 3)
    )

    asset_columns = list(data.columns)
    portfolio_returns = pd.Series(index=data.index, dtype=float)
    weights_history: Dict[pd.Timestamp, pd.Series] = {}
    turnover = pd.Series(dtype=float)
    per_period_turnover = pd.Series(0.0, index=data.index, dtype=float)
    tx_costs = pd.Series(dtype=float)
    training_windows: Dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp]] = {}

    prev_weights = _initial_weights(asset_columns, initial_weights)
    data_values = data.values

    eligible_dates = [date for date in calendar if len(data.loc[:date]) >= window_size]
    if not eligible_dates:
        raise ValueError("window_size too large – no eligible rebalance dates")

    for i, date in enumerate(eligible_dates):
        history = data.loc[:date]
        if window_mode == "rolling":
            train_window = history.tail(window_size)
        else:
            train_window = history
        training_windows[date] = (train_window.index[0], train_window.index[-1])

        raw_weights = strategy(train_window)
        proposed = _normalise_weights(raw_weights, asset_columns)

        delta = float((proposed - prev_weights).abs().sum())
        execute = delta >= float(min_trade)
        new_weights = proposed if execute else prev_weights
        applied_turnover = delta if execute else 0.0
        cost = model.apply(applied_turnover)

        weights_history[date] = new_weights
        turnover.loc[date] = applied_turnover
        tx_costs.loc[date] = float(cost)

        prev_weights = new_weights

        start_idx = data.index.get_loc(date)
        if isinstance(start_idx, slice):
            start_idx = start_idx.stop - 1
        next_date = eligible_dates[i + 1] if i + 1 < len(eligible_dates) else None
        if next_date is not None:
            end_idx = data.index.get_loc(next_date)
            if isinstance(end_idx, slice):
                end_idx = end_idx.start
            stop = end_idx + 1
        else:
            stop = len(data.index)

        apply_slice = slice(start_idx + execution_lag, stop)
        if (
            apply_slice.start >= len(data.index)
            or apply_slice.start >= apply_slice.stop
        ):
            continue

        pending_cost = float(cost)
        if 0 <= apply_slice.start < len(per_period_turnover):
            per_period_turnover.iloc[apply_slice.start] = applied_turnover
        for idx in range(apply_slice.start, apply_slice.stop):
            ret = float(np.dot(data_values[idx], prev_weights.values))
            if pending_cost:
                ret -= pending_cost
                pending_cost = 0.0
            portfolio_returns.iloc[idx] = ret

    active_mask = portfolio_returns.notna()
    filled_returns = portfolio_returns.fillna(0.0)
    equity_curve = (1.0 + filled_returns).cumprod()
    drawdown = _compute_drawdown(equity_curve)
    rolling_input = portfolio_returns.where(active_mask)
    rolling_sharpe = _rolling_sharpe(rolling_input, periods_per_year, roll_window)

    metrics = _compute_metrics(
        filled_returns,
        equity_curve,
        drawdown,
        periods_per_year,
        active_mask,
    )

    weights_df = (
        pd.DataFrame(weights_history).T.reindex(columns=asset_columns).fillna(0.0)
        if weights_history
        else pd.DataFrame(columns=asset_columns, dtype=float)
    )
    if not weights_df.empty and execution_lag:
        weights_df = shift_by_execution_lag(weights_df, lag=execution_lag)

    return BacktestResult(
        returns=portfolio_returns,
        equity_curve=equity_curve,
        weights=weights_df,
        turnover=turnover.sort_index(),
        per_period_turnover=per_period_turnover,
        transaction_costs=tx_costs.sort_index(),
        rolling_sharpe=rolling_sharpe,
        drawdown=drawdown,
        metrics=metrics,
        calendar=pd.DatetimeIndex(eligible_dates),
        window_mode=window_mode,
        window_size=window_size,
        training_windows=training_windows,
        cost_model=model,
        execution_lag=execution_lag,
    )


def _prepare_returns(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "returns index must be a DatetimeIndex or include a 'Date' column"
        )
    df = df.sort_index()
    numeric_df = df.select_dtypes(include=["number"]).astype(float)
    if numeric_df.empty:
        raise ValueError("returns must contain numeric columns")
    return numeric_df


def _rebalance_calendar(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    offset = to_offset(_normalise_frequency(freq))
    resampled = index.to_series().resample(offset).last().dropna()
    calendar = pd.DatetimeIndex(resampled, name="rebalance_date")
    return calendar.intersection(index)


def _normalise_frequency(freq: str) -> str:
    freq_clean = freq.strip()
    freq_upper = freq_clean.upper()
    replacements = {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}
    for suffix, replacement in replacements.items():
        if freq_upper.endswith(replacement):
            return freq_clean
        if freq_upper.endswith(suffix):
            prefix = freq_upper[: -len(suffix)]
            if prefix and not prefix.isdigit():
                continue
            return prefix + replacement
    return freq_clean


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    diffs = np.diff(index.values.astype("datetime64[ns]").astype(np.int64))
    if len(diffs) == 0:
        return 1
    median_ns = np.median(diffs)
    if median_ns <= 0:
        return 1
    median_days = median_ns / (24 * 60 * 60 * 1e9)
    approx = int(round(365 / median_days)) if median_days else 1
    if approx >= 300:
        return 252
    if 45 <= approx <= 60:
        return 52
    if 10 <= approx <= 14:
        return 12
    if 3 <= approx <= 5:
        return 4
    return max(1, approx)


def _initial_weights(
    columns: Sequence[str], initial: Mapping[str, float] | None
) -> pd.Series:
    base = pd.Series(0.0, index=columns, dtype=float)
    if initial is None:
        return base
    init = pd.Series(initial, dtype=float)
    return base.add(init, fill_value=0.0)


def _normalise_weights(
    weights: pd.Series | Mapping[str, float],
    columns: Sequence[str],
) -> pd.Series:
    if isinstance(weights, pd.Series):
        series = weights.astype(float)
    else:
        series = pd.Series(dict(weights), dtype=float)
    series = series.reindex(columns, fill_value=0.0)
    return series


def _compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown


def _rolling_sharpe(
    returns: pd.Series, periods_per_year: int, window: int
) -> pd.Series:
    if window <= 1:
        window = 2
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std(ddof=0)
    sharpe = rolling_mean / rolling_std
    sharpe *= np.sqrt(periods_per_year)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def _compute_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    drawdown: pd.Series,
    periods_per_year: int,
    active_mask: pd.Series,
) -> Dict[str, float]:
    active_returns = returns.where(active_mask).dropna()
    if active_returns.empty:
        active_returns = returns.iloc[0:0]

    total_periods = len(active_returns)
    cumulative_return = float(equity_curve.iloc[-1]) if len(equity_curve) else 1.0
    years = total_periods / periods_per_year if periods_per_year else 0.0

    if years > 0 and cumulative_return > 0:
        cagr = cumulative_return ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")

    vol = active_returns.std(ddof=0) * np.sqrt(periods_per_year)
    downside = active_returns[active_returns < 0]
    downside_std = downside.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = (
        active_returns.mean() / active_returns.std(ddof=0) * np.sqrt(periods_per_year)
        if active_returns.std(ddof=0)
        else float("nan")
    )
    sortino = active_returns.mean() / downside_std if downside_std else float("nan")
    max_drawdown = float(drawdown.min()) if len(drawdown) else float("nan")
    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown and not np.isnan(cagr)
        else float("nan")
    )

    return {
        "cagr": _to_float(cagr),
        "volatility": _to_float(vol),
        "sortino": _to_float(sortino),
        "calmar": _to_float(calmar),
        "max_drawdown": _to_float(max_drawdown),
        "final_value": _to_float(cumulative_return),
        "sharpe": _to_float(sharpe),
    }


def _series_to_dict(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {}
    cleaned = series.dropna()
    return {idx.isoformat(): _to_float(val) for idx, val in cleaned.items()}


def _weights_to_dict(weights: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if weights.empty:
        return {}
    result: Dict[str, Dict[str, float]] = {}
    for timestamp, row in weights.dropna(how="all").iterrows():
        cleaned_row = {
            col: _to_float(val)
            for col, val in row.items()
            if not pd.isna(val) and not np.isclose(val, 0.0)
        }
        if cleaned_row:
            result[timestamp.isoformat()] = cleaned_row
    return result


def _json_default(obj: object) -> object:
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _to_float(value: float | np.floating[Any] | np.integer[Any]) -> float:
    return float(value) if value is not None and not pd.isna(value) else float("nan")


__all__ = ["BacktestResult", "run_backtest"]
