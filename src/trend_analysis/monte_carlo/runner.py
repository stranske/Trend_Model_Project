"""Monte Carlo path evaluation helpers that reuse cached score frames."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from .cache import PathContextCache

ScoreFrameFn = Callable[[Hashable], pd.DataFrame]
StrategyFn = Callable[[Mapping[Hashable, pd.DataFrame]], Any]


def evaluate_strategies_for_path(
    path_id: Hashable,
    rebalance_dates: Iterable[Hashable],
    compute_score_frame: ScoreFrameFn,
    strategies: Mapping[str, StrategyFn],
    *,
    columns_by_strategy: Mapping[str, Sequence[str]] | None = None,
    cache: PathContextCache | None = None,
) -> dict[str, Any]:
    """Evaluate strategies for a single path using cached score frames."""
    cache_obj = cache or PathContextCache()
    dates = list(rebalance_dates)
    for date in dates:

        def _compute_for_date(d: Hashable = date) -> pd.DataFrame:
            return compute_score_frame(d)

        cache_obj.get_or_compute_score_frame(
            path_id,
            date,
            _compute_for_date,
        )

    results: dict[str, Any] = {}
    try:
        for name, strategy in strategies.items():
            columns = columns_by_strategy.get(name) if columns_by_strategy else None
            score_frames = {
                date: cache_obj.select_score_frame(path_id, date, columns) for date in dates
            }
            results[name] = strategy(score_frames)
    finally:
        cache_obj.clear(path_id)
    return results


__all__ = ["evaluate_strategies_for_path"]
