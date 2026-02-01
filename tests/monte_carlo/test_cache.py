import pandas as pd
import pytest

from trend_analysis.monte_carlo.cache import PathContext, PathContextCache
from trend_analysis.monte_carlo.runner import evaluate_strategies_for_path


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
