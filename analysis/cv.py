from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from trend_analysis.metrics import max_drawdown, sharpe_ratio
from trend_analysis.walk_forward import infer_periods_per_year

__all__ = ["Report", "walk_forward", "export_report"]


def _frame_to_markdown(df: pd.DataFrame) -> str:
    try:
        markdown = df.to_markdown(index=False)
    except ImportError:
        markdown = df.to_string(index=False)
    return cast(str, markdown)


@dataclass(slots=True)
class Report:
    """Container holding per-fold metrics and the combined OOS summary."""

    folds: pd.DataFrame
    summary: pd.DataFrame
    oos_returns: pd.Series

    def to_markdown(self) -> str:
        lines = ["# Walk-forward cross-validation", ""]
        if not self.summary.empty:
            lines.append("## OOS summary")
            lines.append(_frame_to_markdown(self.summary))
            lines.append("")
        if not self.folds.empty:
            lines.append("## Fold metrics")
            lines.append(_frame_to_markdown(self.folds))
            lines.append("")
        return "\n".join(lines)


def _prepare_frame(data: pd.DataFrame | Mapping[str, Sequence[Any]]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex or a 'Date' column")
    numeric = df.select_dtypes(include=["number"]).astype(float)
    if numeric.empty:
        raise ValueError("data must include at least one numeric column")
    return numeric.sort_index()


def _build_splits(
    index: pd.DatetimeIndex, folds: int, expand: bool
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    if folds <= 0:
        raise ValueError("folds must be positive")
    n = len(index)
    if n < 2:
        raise ValueError("data must contain at least two rows")
    test_size = max(1, n // (folds + 1))
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for i in range(folds):
        train_end = test_size * (i + 1)
        test_end = min(test_size * (i + 2), n)
        if test_end <= train_end:
            break
        if expand:
            train_slice = index[:train_end]
        else:
            train_start = max(0, train_end - test_size)
            train_slice = index[train_start:train_end]
        test_slice = index[train_end:test_end]
        splits.append((train_slice, test_slice))
    if not splits:
        raise ValueError("fold configuration produced zero splits")
    return splits


def _select_weights(
    train_df: pd.DataFrame,
    *,
    top_n: int,
    lookback: int | None,
    rng: np.random.Generator | None,
) -> pd.Series:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    window = train_df.tail(lookback) if lookback else train_df
    scores = window.mean()
    if scores.empty:
        return pd.Series(dtype=float)
    order = scores.sort_values(ascending=False)
    if rng is not None and len(order) > 1:
        jitter = pd.Series(rng.random(len(order)), index=order.index)
        order = (
            pd.DataFrame({"score": order, "jitter": jitter})
            .sort_values(by=["score", "jitter"], ascending=[False, True])
            .index
        )
    else:
        order = order.index
    selected = list(order[:top_n])
    if not selected:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0 / len(selected), index=selected, dtype=float)
    return weights


def _turnover(prev: pd.Series, current: pd.Series) -> float:
    universe = prev.index.union(current.index)
    delta = current.reindex(universe, fill_value=0.0) - prev.reindex(universe, fill_value=0.0)
    return float(delta.abs().sum())


def walk_forward(
    data: pd.DataFrame | Mapping[str, Sequence[Any]],
    *,
    folds: int = 3,
    expand: bool = True,
    params: Mapping[str, Any] | None = None,
) -> Report:
    """Run a simple walk-forward cross-validation sweep.

    Parameters
    ----------
    data:
        DataFrame (or mapping coercible to DataFrame) with a datetime index or
        ``Date`` column and numeric return columns.
    folds:
        Number of splits to evaluate. Each split uses the next block of rows as
        the test window.
    expand:
        Whether to use expanding training windows (if ``True``) or rolling
        windows with constant length (if ``False``).
    params:
        Strategy parameters. Supported keys:

        * ``top_n`` (int): number of assets to hold per fold (default: 3).
        * ``lookback`` (int): trailing rows to compute scores (default: full
          training window).
        * ``cost_per_turnover`` (float): proportional cost per unit turnover
          (default: 0.0).
        * ``seed`` (int): optional RNG seed for tie-breaking.

    Returns
    -------
    Report
        A report containing per-fold metrics and the combined OOS summary.
    """

    df = _prepare_frame(data)
    cfg = dict(params or {})
    top_n = int(cfg.get("top_n", 3))
    lookback = cfg.get("lookback")
    lookback = int(lookback) if lookback is not None else None
    cost_per_turnover = float(cfg.get("cost_per_turnover", 0.0))
    seed = cfg.get("seed")
    rng = np.random.default_rng(seed) if seed is not None else None

    splits = _build_splits(df.index, folds=folds, expand=expand)
    periods_per_year = infer_periods_per_year(df.index)

    prev_weights = pd.Series(dtype=float)
    fold_records: list[dict[str, Any]] = []
    oos_returns: list[pd.Series] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        weights = _select_weights(train_df, top_n=top_n, lookback=lookback, rng=rng)
        turnover = _turnover(prev_weights, weights)
        cost_drag = turnover * cost_per_turnover
        prev_weights = weights

        if test_df.empty:
            net_returns = pd.Series(dtype=float, index=test_idx)
        elif weights.empty:
            net_returns = pd.Series(0.0, index=test_idx, dtype=float)
        else:
            raw = test_df.reindex(columns=weights.index, fill_value=0.0)
            raw_returns = raw.mul(weights, axis=1).sum(axis=1)
            drag = cost_drag / len(raw_returns) if cost_drag else 0.0
            net_returns = raw_returns - drag

        if len(net_returns):
            sharpe = float(
                sharpe_ratio(net_returns, risk_free=0.0, periods_per_year=periods_per_year)
            )
            drawdown = float(max_drawdown(net_returns))
        else:
            sharpe = float("nan")
            drawdown = float("nan")

        record = {
            "fold": fold_idx,
            "train_start": train_idx[0],
            "train_end": train_idx[-1],
            "test_start": test_idx[0],
            "test_end": test_idx[-1],
            "oos_sharpe": sharpe,
            "oos_max_drawdown": drawdown,
            "turnover": turnover,
            "cost_drag": cost_drag,
            "selected": "|".join(weights.index),
            "weights": {k: float(v) for k, v in weights.items()},
        }
        fold_records.append(record)
        oos_returns.append(net_returns)

    folds_df = pd.DataFrame.from_records(fold_records)
    if not folds_df.empty:
        folds_df["train_start"] = pd.to_datetime(folds_df["train_start"])
        folds_df["train_end"] = pd.to_datetime(folds_df["train_end"])
        folds_df["test_start"] = pd.to_datetime(folds_df["test_start"])
        folds_df["test_end"] = pd.to_datetime(folds_df["test_end"])

    combined = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "folds": len(fold_records),
                "oos_sharpe": (
                    float(sharpe_ratio(combined, risk_free=0.0, periods_per_year=periods_per_year))
                    if len(combined)
                    else float("nan")
                ),
                "oos_max_drawdown": (
                    float(max_drawdown(combined)) if len(combined) else float("nan")
                ),
                "avg_turnover": (
                    float(folds_df["turnover"].mean()) if not folds_df.empty else float("nan")
                ),
                "total_cost_drag": (
                    float(folds_df["cost_drag"].sum()) if not folds_df.empty else 0.0
                ),
            }
        ]
    )
    return Report(folds=folds_df, summary=summary, oos_returns=combined)


def export_report(report: Report, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = output_dir / "cv_folds.csv"
    summary_path = output_dir / "cv_summary.csv"
    markdown_path = output_dir / "cv_report.md"

    report.folds.to_csv(folds_path, index=False)
    report.summary.to_csv(summary_path, index=False)
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")

    return {"folds": folds_path, "summary": summary_path, "markdown": markdown_path}
