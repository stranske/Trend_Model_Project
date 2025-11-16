from __future__ import annotations

import numpy as np
import pandas as pd

from trend_analysis.walk_forward import (
    StrategyConfig,
    WindowConfig,
    evaluate_parameter_grid,
)


class _DummyRng:
    def __init__(self, values: list[float]):
        self.values = list(values)

    def random(self, size: int) -> np.ndarray:
        if len(self.values) < size:
            raise ValueError("insufficient tie-breaker values")
        out = np.array(self.values[:size], dtype=float)
        self.values = self.values[size:]
        return out


def _demo_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    data = {
        "A": np.linspace(0.01, 0.02, len(dates)),
        "B": np.linspace(-0.01, 0.015, len(dates)),
        "C": np.linspace(0.005, 0.025, len(dates)),
    }
    return pd.DataFrame(data, index=dates)


def test_evaluate_parameter_grid_produces_fold_metrics() -> None:
    returns = _demo_returns()
    windows = WindowConfig(train=6, test=3, step=3)
    strategy = StrategyConfig(
        top_n=2,
        defaults={"band": -1.0},
        grid={"lookback": [3]},
    )

    folds, summary = evaluate_parameter_grid(returns, windows, strategy)

    assert len(folds) == 2
    assert set(summary.columns).issuperset({
        "param_lookback",
        "param_band",
        "mean_cagr",
        "folds",
    })
    assert summary["folds"].iloc[0] == 2
    assert summary["mean_cagr"].iloc[0] == summary["mean_cagr"].iloc[0]


def test_seed_changes_tie_breaking_between_identical_scores() -> None:
    returns = _demo_returns()
    returns.loc[:, :] = 0.01  # identical returns across managers
    windows = WindowConfig(train=6, test=3, step=3)
    strategy = StrategyConfig(top_n=1, grid={"lookback": [3], "band": [-1.0]})

    folds_a, _ = evaluate_parameter_grid(
        returns, windows, strategy, rng=_DummyRng([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    )
    folds_b, _ = evaluate_parameter_grid(
        returns, windows, strategy, rng=_DummyRng([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    )

    selected_a = folds_a.loc[0, "selected"]
    selected_b = folds_b.loc[0, "selected"]
    assert selected_a != selected_b
