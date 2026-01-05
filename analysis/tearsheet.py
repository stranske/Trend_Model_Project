"""Generate a lightweight portfolio tearsheet."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
import numpy as np
import pandas as pd

from .results import Results, build_metadata

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402  (deferred import for Agg)
from matplotlib.axes import Axes

DEFAULT_LAST_RUN = Path("demo/portfolio_test_results/last_run_results.json")
DEFAULT_OUTPUT = Path("reports/tearsheet.md")
_ROOT = Path(__file__).resolve().parents[1]


def _ensure_datetime_index(series: pd.Series) -> pd.Series:
    if isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return series
    try:
        converted = pd.DatetimeIndex(pd.to_datetime(series.index))
    except (TypeError, ValueError):
        return series
    return pd.Series(series.values, index=converted, dtype=float)


def _periods_per_year(index: pd.Index) -> float:
    if len(index) < 2:
        return 1.0
    if isinstance(index, pd.PeriodIndex):
        diffs = index.to_timestamp().to_series().diff().dropna()
    elif isinstance(index, pd.DatetimeIndex):
        diffs = index.to_series().diff().dropna()
    else:
        diffs = pd.Series(index).diff().dropna()
    if diffs.empty:
        return 1.0
    median_days = diffs.median() / pd.Timedelta(days=1)
    if median_days <= 0:
        return 1.0
    return float(365.25 / median_days)


def _annualised_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty:
        return float("nan")
    vol = returns.std(ddof=0)
    if vol == 0:
        return float("nan")
    return float((returns.mean() * periods_per_year) / (vol * math.sqrt(periods_per_year)))


def _annualised_volatility(returns: pd.Series, periods_per_year: float) -> float:
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=0) * math.sqrt(periods_per_year))


def _cagr(equity: pd.Series, periods_per_year: float) -> float:
    if equity.empty:
        return float("nan")
    total_return = equity.iloc[-1]
    if total_return <= 0:
        return float("nan")
    years = len(equity) / periods_per_year if periods_per_year else 0
    if years == 0:
        return float("nan")
    return float(total_return ** (1 / years) - 1)


def _serialise_results(results: Results) -> dict[str, Any]:
    return {
        "returns": {str(idx): float(val) for idx, val in results.returns.items()},
        "weights": {str(idx): float(val) for idx, val in results.weights.items()},
        "exposures": {str(idx): float(val) for idx, val in results.exposures.items()},
        "turnover": {str(idx): float(val) for idx, val in results.turnover.items()},
        "costs": {str(k): float(v) for k, v in results.costs.items()},
        "metadata": results.metadata,
    }


def _inflate_results(payload: Mapping[str, Any]) -> Results:
    if "returns" in payload:
        returns = pd.Series(payload.get("returns", {}), dtype=float)
        exposures = pd.Series(payload.get("exposures", {}), dtype=float)
        turnover = pd.Series(payload.get("turnover", {}), dtype=float)
        weights = pd.Series(payload.get("weights", {}), dtype=float)
        costs = payload.get("costs") or {}
        metadata = payload.get("metadata") or {}
        return Results(
            returns=_ensure_datetime_index(returns).sort_index(),
            weights=weights,
            exposures=exposures,
            turnover=_ensure_datetime_index(turnover).sort_index(),
            costs=costs,
            metadata=metadata,
        )
    return Results.from_payload(payload)


def bootstrap_demo_results(
    data_path: Path | str = "demo/extended_returns.csv",
) -> Results:
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = (_ROOT / data_path).resolve()
    frame = pd.read_csv(data_path)
    if "Date" not in frame.columns:
        raise ValueError("Expected a 'Date' column in the demo dataset")
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("Demo dataset contains no numeric return columns")
    returns = numeric.mean(axis=1).astype(float)
    weights = pd.Series(1 / numeric.shape[1], index=numeric.columns, dtype=float)
    turnover = pd.Series(0.0, index=returns.index, dtype=float)
    meta = build_metadata(
        universe=list(numeric.columns),
        lookbacks={
            "in_start": str(returns.index.min().date()),
            "in_end": str(returns.index.max().date()),
            "out_start": str(returns.index.min().date()),
            "out_end": str(returns.index.max().date()),
        },
        costs={"monthly_cost": 0.0},
        selected=list(numeric.columns),
        data_path=data_path,
    )
    return Results(
        returns=returns,
        weights=weights,
        exposures=weights,
        turnover=turnover,
        costs={"monthly_cost": 0.0},
        metadata=meta,
    )


def load_results_payload(path: Path | str = DEFAULT_LAST_RUN) -> Results:
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _inflate_results(payload)
    results = bootstrap_demo_results()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialise_results(results), indent=2), encoding="utf-8")
    return results


def _plot_axes(
    ax: Axes,
    index: Iterable[Any],
    series: Iterable[float],
    title: str,
    *,
    color: str = "tab:blue",
) -> None:
    x_values = np.asarray(list(index))
    y_values = np.asarray(list(series), dtype=float)
    ax.plot(x_values, y_values, color=color)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6)


def render(results: Results, out: Path | str = DEFAULT_OUTPUT) -> tuple[Path, Path]:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    returns = _ensure_datetime_index(results.returns).sort_index()
    turnover = _ensure_datetime_index(results.turnover).reindex_like(returns, method="ffill")

    equity = (1 + returns.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    periods_per_year = _periods_per_year(returns.index)
    sharpe = _annualised_sharpe(returns, periods_per_year)
    vol = _annualised_volatility(returns, periods_per_year)
    cagr = _cagr(equity, periods_per_year)
    max_dd = float(drawdown.min()) if not drawdown.empty else float("nan")
    avg_turnover = float(turnover.mean()) if not turnover.empty else 0.0
    cost_drag = float(
        results.costs.get("turnover_applied") or results.costs.get("monthly_cost") or 0.0
    )

    roll_window = max(6, min(len(returns), 12))
    rolling_sharpe = returns.rolling(roll_window).apply(
        lambda x: _annualised_sharpe(x, periods_per_year), raw=False
    )
    rolling_vol = returns.rolling(roll_window).apply(
        lambda x: _annualised_volatility(x, periods_per_year), raw=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_axes(axes[0, 0], equity.index, equity, "Equity curve", color="tab:green")
    _plot_axes(
        axes[0, 1],
        rolling_sharpe.index,
        rolling_sharpe,
        "Rolling Sharpe",
        color="tab:purple",
    )
    _plot_axes(axes[1, 0], drawdown.index, drawdown, "Drawdown", color="tab:red")
    _plot_axes(axes[1, 1], turnover.index, turnover, "Turnover", color="tab:orange")
    axes[0, 1].plot(
        rolling_vol.index,
        rolling_vol,
        color="tab:blue",
        linestyle="--",
        label="Rolling vol",
    )
    axes[0, 1].legend()
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=20)
    plot_path = out_path.with_suffix(".png")
    fig.savefig(plot_path)
    plt.close(fig)

    stats = pd.DataFrame(
        {
            "Sharpe": sharpe,
            "Volatility": vol,
            "Max drawdown": max_dd,
            "CAGR": cagr,
            "Turnover": avg_turnover,
            "Cost drag": cost_drag,
        },
        index=["value"],
    ).T
    try:
        stats_block = stats.to_markdown()
    except ImportError:
        stats_block = stats.to_string()

    lines = ["# Portfolio tearsheet", "", "## Headline statistics", "", stats_block, ""]
    fingerprint = results.fingerprint()
    if fingerprint:
        lines.extend(["### Run fingerprint", f"- `{fingerprint}`", ""])
    lines.extend(["## Charts", "", f"![Tearsheet charts]({plot_path.name})", ""])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path, plot_path
