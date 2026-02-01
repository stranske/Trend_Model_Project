import time

import pandas as pd
import pytest

from trend_analysis.monte_carlo.cache import PathContext, PathContextCache
from trend_analysis.monte_carlo.runner import evaluate_strategies_for_path


def _evaluate_strategies_naively(
    rebalance_dates: list[str],
    compute_score_frame,
    strategies: dict[str, callable],
    *,
    columns_by_strategy: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for name, strategy in strategies.items():
        columns = columns_by_strategy.get(name) if columns_by_strategy else None
        frames: dict[str, pd.DataFrame] = {}
        for date in rebalance_dates:
            frame = compute_score_frame(date)
            if columns:
                keep = [col for col in columns if col in frame.columns]
                frame = frame.loc[:, keep].copy()
            frames[date] = frame
        results[name] = strategy(frames)
    return results


def test_path_context_get_or_compute_caches_once() -> None:
    context = PathContext(path_id="path-1")
    calls: list[str] = []

    def compute() -> pd.DataFrame:
        calls.append("called")
        return pd.DataFrame({"Sharpe": [1.0, 2.0]}, index=["A", "B"])

    first = context.get_or_compute("2024-01-31", compute)
    second = context.get_or_compute("2024-01-31", compute)

    assert calls == ["called"]
    assert first is second
    assert context.has_score_frame("2024-01-31")


def test_cache_reuses_score_frames_across_strategies_and_clears() -> None:
    cache = PathContextCache()
    calls: list[str] = []

    def compute_score_frame(date: str) -> pd.DataFrame:
        calls.append(date)
        return pd.DataFrame(
            {
                "Sharpe": [1.0, 2.0],
                "Volatility": [0.1, 0.2],
            },
            index=["A", "B"],
        )

    def sharpe_strategy(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
        for frame in frames.values():
            assert set(frame.columns) == {"Sharpe"}
        return {date: float(frame["Sharpe"].sum()) for date, frame in frames.items()}

    def vol_strategy(frames: dict[str, pd.DataFrame]) -> dict[str, float]:
        for frame in frames.values():
            assert set(frame.columns) == {"Volatility"}
        return {date: float(frame["Volatility"].sum()) for date, frame in frames.items()}

    rebalance_dates = ["2024-01-31", "2024-02-29"]
    results = evaluate_strategies_for_path(
        "path-42",
        rebalance_dates,
        compute_score_frame,
        {"sharpe": sharpe_strategy, "vol": vol_strategy},
        columns_by_strategy={"sharpe": ["Sharpe"], "vol": ["Volatility"]},
        cache=cache,
    )

    assert calls == rebalance_dates
    assert results["sharpe"]["2024-01-31"] == 3.0
    assert results["vol"]["2024-02-29"] == pytest.approx(0.3)
    assert not cache.has_path("path-42")


def test_cache_computes_once_per_date_with_many_strategies() -> None:
    cache = PathContextCache()
    calls: list[str] = []

    def compute_score_frame(date: str) -> pd.DataFrame:
        calls.append(date)
        return pd.DataFrame(
            {
                "Sharpe": [1.0, 2.0],
                "Volatility": [0.1, 0.2],
                "Return": [0.05, 0.07],
            },
            index=["A", "B"],
        )

    def make_strategy(column: str, idx: int) -> tuple[str, callable]:
        def strategy(frames: dict[str, pd.DataFrame]) -> float:
            return float(sum(frame[column].sum() for frame in frames.values()))

        return f"{column}-strategy-{idx}", strategy

    columns = [
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
    ]
    strategies = dict(
        [make_strategy(column, idx) for idx, column in enumerate(columns, start=1)],
    )
    columns_by_strategy = {name: [name.split("-")[0]] for name in strategies}

    rebalance_dates = ["2024-01-31", "2024-02-29", "2024-03-29"]
    results = evaluate_strategies_for_path(
        "path-99",
        rebalance_dates,
        compute_score_frame,
        strategies,
        columns_by_strategy=columns_by_strategy,
        cache=cache,
    )

    assert calls == rebalance_dates
    assert len(results) == 10
    assert not cache.has_path("path-99")


def test_cached_results_match_naive_evaluation() -> None:
    cache = PathContextCache()

    def compute_score_frame(date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Sharpe": [1.0, 2.0],
                "Volatility": [0.1, 0.2],
                "Return": [0.05, 0.07],
            },
            index=["A", "B"],
        )

    def sharpe_strategy(frames: dict[str, pd.DataFrame]) -> float:
        return float(sum(frame["Sharpe"].sum() for frame in frames.values()))

    def vol_strategy(frames: dict[str, pd.DataFrame]) -> float:
        return float(sum(frame["Volatility"].sum() for frame in frames.values()))

    rebalance_dates = ["2024-01-31", "2024-02-29", "2024-03-29"]
    strategies = {"sharpe": sharpe_strategy, "vol": vol_strategy}
    columns_by_strategy = {"sharpe": ["Sharpe"], "vol": ["Volatility"]}

    cached_results = evaluate_strategies_for_path(
        "path-compare",
        rebalance_dates,
        compute_score_frame,
        strategies,
        columns_by_strategy=columns_by_strategy,
        cache=cache,
    )
    naive_results = _evaluate_strategies_naively(
        rebalance_dates,
        compute_score_frame,
        strategies,
        columns_by_strategy=columns_by_strategy,
    )

    assert cached_results == naive_results
    assert not cache.has_path("path-compare")


@pytest.mark.performance
def test_cached_strategies_are_materially_faster() -> None:
    base_frame = pd.DataFrame(
        {
            "Sharpe": [1.0, 2.0],
            "Volatility": [0.1, 0.2],
            "Return": [0.05, 0.07],
        },
        index=["A", "B"],
    )

    def make_compute(delay: float):
        def compute_score_frame(date: str) -> pd.DataFrame:  # noqa: ARG001
            time.sleep(delay)
            return base_frame.copy()

        return compute_score_frame

    def make_strategy(column: str, idx: int) -> tuple[str, callable]:
        def strategy(frames: dict[str, pd.DataFrame]) -> float:
            return float(sum(frame[column].sum() for frame in frames.values()))

        return f"{column}-strategy-{idx}", strategy

    columns = [
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
        "Volatility",
        "Return",
        "Sharpe",
    ]
    strategies = dict(
        [make_strategy(column, idx) for idx, column in enumerate(columns, start=1)],
    )
    columns_by_strategy = {name: [name.split("-")[0]] for name in strategies}
    rebalance_dates = ["2024-01-31", "2024-02-29", "2024-03-29"]

    cached_compute = make_compute(0.005)
    naive_compute = make_compute(0.005)
    single_compute = make_compute(0.005)

    start = time.perf_counter()
    evaluate_strategies_for_path(
        "path-perf",
        rebalance_dates,
        cached_compute,
        strategies,
        columns_by_strategy=columns_by_strategy,
        cache=PathContextCache(),
    )
    cached_time = time.perf_counter() - start

    single_strategy_name = next(iter(strategies))
    single_strategy = {single_strategy_name: strategies[single_strategy_name]}
    single_columns = {single_strategy_name: columns_by_strategy[single_strategy_name]}

    start = time.perf_counter()
    evaluate_strategies_for_path(
        "path-perf-single",
        rebalance_dates,
        single_compute,
        single_strategy,
        columns_by_strategy=single_columns,
        cache=PathContextCache(),
    )
    single_time = time.perf_counter() - start

    start = time.perf_counter()
    _evaluate_strategies_naively(
        rebalance_dates,
        naive_compute,
        strategies,
        columns_by_strategy=columns_by_strategy,
    )
    naive_time = time.perf_counter() - start

    assert cached_time < single_time * 2
    assert cached_time * 2 < naive_time
