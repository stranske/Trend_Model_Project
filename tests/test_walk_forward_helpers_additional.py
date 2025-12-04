import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trend_analysis.walk_forward as wf


@pytest.fixture
def simple_index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=5, freq="D")


def test_window_splits_generates_expected_pairs(simple_index: pd.DatetimeIndex) -> None:
    cfg = wf.WindowConfig(train=2, test=2, step=1)

    splits = wf._window_splits(simple_index, cfg)

    assert [(list(train), list(test)) for train, test in splits] == [
        ([simple_index[0], simple_index[1]], [simple_index[2], simple_index[3]]),
        ([simple_index[1], simple_index[2]], [simple_index[3], simple_index[4]]),
    ]


def test_window_splits_rejects_empty_result(simple_index: pd.DatetimeIndex) -> None:
    cfg = wf.WindowConfig(train=4, test=2, step=1)

    with pytest.raises(ValueError, match="zero walk-forward splits"):
        wf._window_splits(simple_index[:4], cfg)


def test_parameter_grid_orders_keys_and_combinations() -> None:
    strategy = wf.StrategyConfig(grid={"beta": [2, 3], "alpha": [1]})

    combos = wf._parameter_grid(strategy)

    assert combos == [{"alpha": 1, "beta": 2}, {"alpha": 1, "beta": 3}]


def test_tie_breaker_is_deterministic_with_rng(simple_index: pd.DatetimeIndex) -> None:
    rng = np.random.default_rng(123)

    tie_series = wf._tie_breaker(simple_index, rng)

    assert tie_series.index.equals(simple_index)
    assert tie_series.dtype == float
    assert tie_series.tolist() == pytest.approx(
        list(np.array([0.68235186, 0.05382102, 0.22035987, 0.18437181, 0.1759059]))
    )


def test_select_weights_applies_band_and_tie_breaking(
    simple_index: pd.DatetimeIndex,
) -> None:
    data = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, 1.0, 1.0],
            "b": [1.0, 1.0, 1.0, 1.0, 1.0],
            "c": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=simple_index,
    )
    rng = np.random.default_rng(42)
    params = {"top_n": 2, "band": 0.5, "lookback": 3}

    weights = wf._select_weights(data, params, rng)

    assert pytest.approx(weights.sum()) == 1.0
    # With all scores equal, tie-breaker ordering should be deterministic with RNG
    assert list(weights.index) == ["b", "a"]


def test_compute_turnover_handles_union_of_indices() -> None:
    prev = pd.Series({"a": 0.5, "b": 0.5})
    new = pd.Series({"b": 0.5, "c": 0.5})

    turnover = wf._compute_turnover(prev, new)

    assert turnover == pytest.approx(1.0)


def test_evaluate_parameter_grid_builds_records_and_summary(tmp_path: Path) -> None:
    returns = pd.DataFrame(
        {
            "a": [0.01, -0.02, 0.03, 0.04],
            "b": [0.02, 0.01, -0.01, 0.00],
        },
        index=pd.date_range("2021-01-01", periods=4, freq="W"),
    )
    windows = wf.WindowConfig(train=2, test=1, step=1)
    strategy = wf.StrategyConfig(grid={"lookback": [2], "band": [0.0]}, top_n=1)
    rng = np.random.default_rng(7)

    folds, summary = wf.evaluate_parameter_grid(returns, windows, strategy, rng=rng)

    assert not folds.empty
    assert not summary.empty
    assert set(summary.columns).issuperset({"mean_cagr", "mean_sharpe", "folds"})
    assert folds["fold"].iloc[0] == 1


def test_json_default_serializes_known_types(simple_index: pd.DatetimeIndex) -> None:
    payload = {
        "float": np.float64(1.23),
        "int": np.int64(4),
        "timestamp": simple_index[0],
        "series": pd.Series([1, 2]),
    }

    encoded = json.dumps(payload, default=wf._json_default)

    parsed = json.loads(encoded)
    assert parsed == {
        "float": 1.23,
        "int": 4,
        "timestamp": simple_index[0].isoformat(),
        "series": [1, 2],
    }

    with pytest.raises(TypeError):
        wf._json_default(object())
