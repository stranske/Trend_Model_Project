"""Caching helpers for Monte Carlo path evaluation."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence

import pandas as pd


class PathContext:
    """Cache score frames for a single simulated path."""

    def __init__(self, path_id: Hashable | None = None) -> None:
        self.path_id = path_id
        self._score_frames: dict[Hashable, pd.DataFrame] = {}

    def has_score_frame(self, rebalance_date: Hashable) -> bool:
        return rebalance_date in self._score_frames

    def get_score_frame(self, rebalance_date: Hashable) -> pd.DataFrame | None:
        return self._score_frames.get(rebalance_date)

    def set_score_frame(self, rebalance_date: Hashable, score_frame: pd.DataFrame) -> None:
        if not isinstance(score_frame, pd.DataFrame):
            raise TypeError("score_frame must be a pandas DataFrame")
        self._score_frames[rebalance_date] = score_frame

    def get_or_compute(
        self,
        rebalance_date: Hashable,
        compute_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        cached = self._score_frames.get(rebalance_date)
        if cached is not None:
            return cached
        score_frame = compute_fn()
        if not isinstance(score_frame, pd.DataFrame):
            raise TypeError("compute_fn must return a pandas DataFrame")
        self._score_frames[rebalance_date] = score_frame
        return score_frame

    def select_columns(
        self,
        rebalance_date: Hashable,
        columns: Sequence[str] | None,
    ) -> pd.DataFrame:
        score_frame = self._score_frames.get(rebalance_date)
        if score_frame is None:
            raise KeyError(f"No cached score frame for {rebalance_date!r}")
        if not columns:
            return score_frame
        keep = [col for col in columns if col in score_frame.columns]
        return score_frame.loc[:, keep].copy()

    def dates(self) -> Iterable[Hashable]:
        return self._score_frames.keys()

    def clear(self) -> None:
        self._score_frames.clear()


class PathContextCache:
    """Manage cached score frames across Monte Carlo paths."""

    def __init__(self) -> None:
        self._contexts: dict[Hashable, PathContext] = {}

    def get_context(self, path_id: Hashable) -> PathContext:
        context = self._contexts.get(path_id)
        if context is None:
            context = PathContext(path_id=path_id)
            self._contexts[path_id] = context
        return context

    def has_path(self, path_id: Hashable) -> bool:
        return path_id in self._contexts

    def path_ids(self) -> Iterable[Hashable]:
        return self._contexts.keys()

    def get_or_compute_score_frame(
        self,
        path_id: Hashable,
        rebalance_date: Hashable,
        compute_fn: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        context = self.get_context(path_id)
        return context.get_or_compute(rebalance_date, compute_fn)

    def select_score_frame(
        self,
        path_id: Hashable,
        rebalance_date: Hashable,
        columns: Sequence[str] | None,
    ) -> pd.DataFrame:
        context = self.get_context(path_id)
        return context.select_columns(rebalance_date, columns)

    def clear(self, path_id: Hashable | None = None) -> None:
        if path_id is None:
            for context in self._contexts.values():
                context.clear()
            self._contexts.clear()
            return
        context = self._contexts.pop(path_id, None)
        if context is not None:
            context.clear()


__all__ = ["PathContext", "PathContextCache"]
