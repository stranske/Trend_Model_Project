"""Utility helpers for deterministic walk-forward parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from trend_analysis.metrics import annual_return, max_drawdown, sharpe_ratio


@dataclass(slots=True)
class DataConfig:
    csv_path: Path
    date_column: str = "Date"
    columns: Sequence[str] | None = None


@dataclass(slots=True)
class WindowConfig:
    train: int
    test: int
    step: int


@dataclass(slots=True)
class StrategyConfig:
    top_n: int = 5
    defaults: Mapping[str, float | int] = field(default_factory=dict)
    grid: Mapping[str, Sequence[float | int]] = field(default_factory=dict)

    def base_params(self) -> dict[str, float | int]:
        params = {"top_n": self.top_n, "band": 0.0}
        params.update(self.defaults)
        return params


@dataclass(slots=True)
class RunConfig:
    name: str = "wf"
    output_dir: Path = Path("perf/wf")
    seed: int | None = None


@dataclass(slots=True)
class WalkForwardSettings:
    data: DataConfig
    windows: WindowConfig
    strategy: StrategyConfig
    run: RunConfig


def load_settings(path: Path | str) -> WalkForwardSettings:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    data_section = raw.get("data", {})
    wf_section = raw.get("walk_forward", {})
    strat_section = raw.get("strategy", {})
    run_section = raw.get("run", {})

    csv_path = Path(data_section.get("csv_path", ""))
    if not csv_path:
        raise ValueError("data.csv_path must be provided")
    if not csv_path.is_absolute():
        csv_path = (cfg_path.parent / csv_path).resolve()
    date_column = str(data_section.get("date_column", "Date"))
    columns = data_section.get("columns")
    if columns is not None and (
        not isinstance(columns, Sequence) or isinstance(columns, (str, bytes))
    ):
        raise ValueError("data.columns must be a list of column names if provided")

    train = int(wf_section.get("train", 0))
    test = int(wf_section.get("test", 0))
    step = int(wf_section.get("step", 0))
    if min(train, test, step) <= 0:
        raise ValueError("walk_forward.train/test/step must be positive integers")

    top_n = int(strat_section.get("top_n", 5))
    if top_n <= 0:
        raise ValueError("strategy.top_n must be a positive integer")

    defaults = strat_section.get("defaults", {}) or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("strategy.defaults must be a mapping")

    grid = strat_section.get("grid", {}) or {}
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("strategy.grid must contain at least one parameter list")
    prepared_grid: dict[str, list[float | int]] = {}
    for key, values in grid.items():
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            seq = list(values)
        else:
            raise ValueError("strategy.grid values must be sequences")
        if not seq:
            raise ValueError(f"strategy.grid entry '{key}' must contain at least one value")
        prepared_grid[key] = seq

    run_name = str(run_section.get("name", "wf"))
    output_dir = Path(run_section.get("output_dir", "perf/wf"))
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()
    seed_value = run_section.get("seed")
    seed = int(seed_value) if seed_value is not None else None

    return WalkForwardSettings(
        data=DataConfig(csv_path=csv_path, date_column=date_column, columns=columns),
        windows=WindowConfig(train=train, test=test, step=step),
        strategy=StrategyConfig(top_n=top_n, defaults=defaults, grid=prepared_grid),
        run=RunConfig(name=run_name, output_dir=output_dir, seed=seed),
    )


def load_returns(data_cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(data_cfg.csv_path)
    if data_cfg.date_column not in df.columns:
        raise ValueError(f"Date column '{data_cfg.date_column}' not found in CSV")
    df[data_cfg.date_column] = pd.to_datetime(df[data_cfg.date_column])
    df = df.sort_values(data_cfg.date_column)
    df = df.set_index(data_cfg.date_column)
    numeric = df.select_dtypes(include=["number"]).astype(float)
    if numeric.empty:
        raise ValueError("No numeric columns found in returns file")
    if data_cfg.columns:
        missing = [col for col in data_cfg.columns if col not in numeric.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {', '.join(missing)}")
        numeric = numeric.loc[:, list(data_cfg.columns)]
    return numeric


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    diffs = np.diff(index.view("int64"))
    if len(diffs) == 0:
        return 1
    median_ns = np.median(diffs)
    if median_ns <= 0:
        return 1
    median_days = median_ns / (24 * 60 * 60 * 1e9)
    if median_days <= 0:
        return 1
    approx = int(round(365 / median_days))
    if approx >= 300:
        return 252
    if 45 <= approx <= 60:
        return 52
    if 10 <= approx <= 14:
        return 12
    if 3 <= approx <= 5:
        return 4
    if approx <= 0:
        return 1
    return approx


def _window_splits(index: pd.DatetimeIndex, cfg: WindowConfig) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    start = 0
    total = len(index)
    while start + cfg.train + cfg.test <= total:
        train_idx = index[start : start + cfg.train]
        test_idx = index[start + cfg.train : start + cfg.train + cfg.test]
        splits.append((train_idx, test_idx))
        start += cfg.step
    if not splits:
        raise ValueError("Window configuration produced zero walk-forward splits")
    return splits


def _parameter_grid(strategy: StrategyConfig) -> list[dict[str, float | int]]:
    keys = sorted(strategy.grid)
    values = [strategy.grid[k] for k in keys]
    combos: list[dict[str, float | int]] = []
    for product_values in itertools.product(*values):
        combos.append(dict(zip(keys, product_values)))
    return combos


def _tie_breaker(index: pd.Index, rng: np.random.Generator | None) -> pd.Series:
    if rng is None:
        return pd.Series(np.arange(len(index)), index=index, dtype=float)
    return pd.Series(rng.random(len(index)), index=index, dtype=float)


def _select_weights(
    train_df: pd.DataFrame,
    params: Mapping[str, float | int],
    rng: np.random.Generator | None,
) -> pd.Series:
    lookback = int(params.get("lookback", len(train_df)))
    lookback = max(1, min(lookback, len(train_df)))
    band = float(params.get("band", 0.0))
    top_n = max(1, int(params.get("top_n", 1)))

    window = train_df.tail(lookback)
    scores = window.mean()
    eligible = scores[scores >= band]
    ordered_idx: pd.Index
    if eligible.empty:
        ordered_idx = scores.sort_values(ascending=False).index
    else:
        ordered_idx = eligible.sort_values(ascending=False).index
    if rng is not None and len(ordered_idx) > 1:
        ties = scores.loc[ordered_idx]
        order = (
            pd.DataFrame({"score": ties, "tie": _tie_breaker(ordered_idx, rng)})
            .sort_values(by=["score", "tie"], ascending=[False, True])
            .index
        )
        ordered_idx = order
    selected = list(ordered_idx[:top_n])
    if not selected:
        return pd.Series(dtype=float)
    weight = 1.0 / len(selected)
    weights = pd.Series(weight, index=selected, dtype=float)
    return weights


def _compute_turnover(prev: pd.Series, new: pd.Series) -> float:
    union = prev.index.union(new.index)
    delta = new.reindex(union, fill_value=0.0) - prev.reindex(union, fill_value=0.0)
    return float(delta.abs().sum())


def evaluate_parameter_grid(
    returns: pd.DataFrame,
    windows: WindowConfig,
    strategy: StrategyConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if returns.empty:
        raise ValueError("returns DataFrame must not be empty")
    splits = _window_splits(returns.index, windows)
    combos = _parameter_grid(strategy)
    base_params = strategy.base_params()
    periods_per_year = infer_periods_per_year(returns.index)

    records: list[dict[str, Any]] = []

    for combo in combos:
        params = base_params | combo
        prev_weights = pd.Series(dtype=float)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            train_df = returns.loc[train_idx]
            test_df = returns.loc[test_idx]
            weights = _select_weights(train_df, params, rng)
            turnover = _compute_turnover(prev_weights, weights)
            prev_weights = weights

            if weights.empty or test_df.empty:
                fold_returns = pd.Series(0.0, index=test_idx, dtype=float)
            else:
                subset = test_df.reindex(columns=weights.index, fill_value=0.0)
                fold_returns = subset.mul(weights, axis=1).sum(axis=1)

            cagr = annual_return(fold_returns, periods_per_year=periods_per_year)
            sharpe = sharpe_ratio(
                fold_returns, risk_free=0.0, periods_per_year=periods_per_year
            )
            drawdown = max_drawdown(fold_returns)
            hit_rate = float(fold_returns.gt(0).mean()) if len(fold_returns) else np.nan

            record = {
                "fold": fold_idx,
                "train_start": train_idx[0],
                "train_end": train_idx[-1],
                "test_start": test_idx[0],
                "test_end": test_idx[-1],
                "cagr": float(cagr),
                "sharpe": float(sharpe),
                "max_drawdown": float(drawdown),
                "hit_rate": hit_rate,
                "turnover": turnover,
                "selected": "|".join(weights.index) if len(weights) else "",
            }
            for key, value in params.items():
                record[f"param_{key}"] = value
            records.append(record)

    folds_df = pd.DataFrame.from_records(records)
    param_cols = [col for col in folds_df.columns if col.startswith("param_")]
    agg_map = {
        "cagr": "mean",
        "sharpe": "mean",
        "max_drawdown": "mean",
        "hit_rate": "mean",
        "turnover": "mean",
    }
    summary = (
        folds_df.groupby(param_cols, dropna=False)
        .agg({**agg_map, "fold": "count"})
        .rename(columns={"fold": "folds"})
        .reset_index()
    )
    summary = summary.rename(
        columns={
            "cagr": "mean_cagr",
            "sharpe": "mean_sharpe",
            "max_drawdown": "mean_max_drawdown",
            "hit_rate": "mean_hit_rate",
            "turnover": "mean_turnover",
        }
    )
    return folds_df, summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.to_list()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default))
            handle.write("\n")


def _maybe_render_heatmap(summary: pd.DataFrame, output: Path) -> None:
    param_cols = [col for col in summary.columns if col.startswith("param_")]
    varying = [col for col in param_cols if summary[col].nunique() > 1]
    if len(varying) != 2:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return

    pivot = summary.pivot(
        index=varying[0], columns=varying[1], values="mean_cagr"
    ).sort_index(axis=0).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(varying[1])
    ax.set_ylabel(varying[0])
    ax.set_title("Mean CAGR heatmap")
    fig.colorbar(im, ax=ax, label="mean CAGR")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def persist_artifacts(
    settings: WalkForwardSettings,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    config_path: Path,
) -> Path:
    settings.run.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = settings.run.output_dir / f"{settings.run.name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    folds_path = run_dir / "folds.csv"
    summary_path = run_dir / "summary.csv"
    jsonl_path = run_dir / "summary.jsonl"
    heatmap_path = run_dir / "mean_cagr_heatmap.png"

    folds.to_csv(folds_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_jsonl(jsonl_path, summary.to_dict("records"))
    _maybe_render_heatmap(summary, heatmap_path)

    target_cfg = run_dir / "config_used.yml"
    target_cfg.write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    return run_dir


def run_from_config(path: Path | str) -> Path:
    cfg_path = Path(path)
    settings = load_settings(cfg_path)
    returns = load_returns(settings.data)
    rng = (
        np.random.default_rng(settings.run.seed)
        if settings.run.seed is not None
        else None
    )
    folds, summary = evaluate_parameter_grid(
        returns, settings.windows, settings.strategy, rng=rng
    )
    return persist_artifacts(settings, folds, summary, config_path=cfg_path)


__all__ = [
    "DataConfig",
    "WindowConfig",
    "StrategyConfig",
    "RunConfig",
    "WalkForwardSettings",
    "load_settings",
    "load_returns",
    "infer_periods_per_year",
    "evaluate_parameter_grid",
    "persist_artifacts",
    "run_from_config",
]
