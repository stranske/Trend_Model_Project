from __future__ import annotations

import pandas as pd

from .core.rank_selection import ASCENDING_METRICS
from .plugins import Selector, create_selector, selector_registry


@selector_registry.register("rank")
class RankSelector(Selector):
    """Select top N funds based on a metric column."""

    def __init__(self, top_n: int, rank_column: str) -> None:
        self.top_n = top_n
        self.rank_column = rank_column

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.rank_column not in score_frame.columns:
            raise KeyError(self.rank_column)
        scores = score_frame[self.rank_column]
        ascending = self.rank_column in ASCENDING_METRICS
        ranked = scores.rank(ascending=ascending, method="first")
        selected_idx = ranked.nsmallest(self.top_n).index
        selected = score_frame.loc[selected_idx].copy()
        log = pd.DataFrame(
            {
                "candidate": score_frame.index,
                "metric": self.rank_column,
                "reason": ranked,
            }
        ).set_index("candidate")
        return selected, log


@selector_registry.register("zscore")
class ZScoreSelector(Selector):
    """Filter by z-score threshold."""

    def __init__(self, threshold: float, *, direction: int = 1, column: str = "Sharpe") -> None:
        self.threshold = threshold
        self.direction = 1 if direction >= 0 else -1
        self.column = column

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.column not in score_frame.columns:
            raise KeyError(self.column)
        scores = score_frame[self.column].astype(float)
        mu = scores.mean()
        sigma = scores.std(ddof=0)
        z = (scores - mu) / (sigma if sigma else 1.0)
        mask = z * self.direction > self.threshold
        selected = score_frame.loc[mask].copy()
        log = pd.DataFrame(
            {"candidate": score_frame.index, "metric": self.column, "reason": z}
        ).set_index("candidate")
        return selected, log


def create_selector_by_name(name: str, **params: object) -> Selector:
    """Factory helper for creating selectors via the plugin registry."""
    return create_selector(name, **params)


__all__ = [
    "RankSelector",
    "ZScoreSelector",
    "create_selector_by_name",
    "selector_registry",
]
