from __future__ import annotations

from typing import Sequence

import pandas as pd

from .core.rank_selection import ASCENDING_METRICS
from .plugins import Selector, create_selector, selector_registry


@selector_registry.register("rank")
class RankSelector(Selector):
    """Select top N funds based on a metric column."""

    def __init__(
        self,
        top_n: int,
        rank_column: str,
        *,
        tie_breakers: Sequence[str] | None = None,
    ) -> None:
        self.top_n = top_n
        self.rank_column = rank_column
        # Default to MaxDrawdown as a deterministic tie-breaker when available.
        self.tie_breakers = tuple(tie_breakers) if tie_breakers is not None else ("MaxDrawdown",)

    def select(self, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.rank_column not in score_frame.columns:
            raise KeyError(self.rank_column)

        frame = score_frame.copy()
        ascending = self.rank_column in ASCENDING_METRICS
        metric_values = frame[self.rank_column].astype(float)
        fill_primary = float("inf") if ascending else float("-inf")
        order_frame = pd.DataFrame(index=frame.index)
        order_frame["__primary"] = metric_values.fillna(fill_primary)

        sort_cols = ["__primary"]
        sort_orders = [ascending]
        for col in self.tie_breakers:
            if col == self.rank_column or col not in frame.columns:
                continue
            tie_series = frame[col].astype(float)
            tie_fill = float("inf") if col in ASCENDING_METRICS else float("-inf")
            key = f"__tie_{col}"
            order_frame[key] = tie_series.fillna(tie_fill)
            sort_cols.append(key)
            sort_orders.append(col in ASCENDING_METRICS)

        ranked_idx = order_frame.sort_values(
            by=sort_cols, ascending=sort_orders, kind="mergesort"
        ).index
        selected = frame.loc[ranked_idx[: self.top_n]].copy()

        order = pd.Series(range(1, len(ranked_idx) + 1), index=ranked_idx, dtype=float)
        reason = pd.Series(float("nan"), index=score_frame.index, dtype=float)
        reason.loc[order.index] = order
        log = pd.DataFrame(
            {
                "candidate": score_frame.index,
                "metric": self.rank_column,
                "reason": reason,
            }
        ).set_index("candidate")
        return selected, log


@selector_registry.register("zscore")
class ZScoreSelector(Selector):
    """Filter by z-score threshold."""

    def __init__(
        self, threshold: float, *, direction: int = 1, column: str = "Sharpe"
    ) -> None:
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
