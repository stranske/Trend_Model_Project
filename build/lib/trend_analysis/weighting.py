from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._typing import FloatArray


class BaseWeighting:
    """Base interface for weighting schemes."""

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        raise NotImplementedError


class EqualWeight(BaseWeighting):
    """Simple equal weighting."""

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame(columns=["weight"])
        n = len(selected)
        w = np.repeat(1.0 / n, n)
        w = (w * 10000).round() / 10000
        return pd.DataFrame({"weight": w}, index=selected.index)


class ScorePropSimple(BaseWeighting):
    """Weights proportional to positive scores."""

    def __init__(self, column: str = "Sharpe") -> None:
        self.column = column

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame(columns=["weight"])
        if self.column not in selected.columns:
            raise KeyError(self.column)
        scores = selected[self.column].clip(lower=0).astype(float)
        if scores.sum() == 0:
            return EqualWeight().weight(selected)
        w = scores / scores.sum()
        return pd.DataFrame({"weight": w}, index=selected.index)


class ScorePropBayesian(BaseWeighting):
    """Score-proportional weighting with Bayesian shrinkage."""

    def __init__(self, column: str = "Sharpe", *, shrink_tau: float = 0.25) -> None:
        self.column = column
        self.tau = float(shrink_tau)

    def weight(self, selected: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        if selected.empty:
            return pd.DataFrame(columns=["weight"])
        if self.column not in selected.columns:
            raise KeyError(self.column)
        scores = selected[self.column].astype(float)
        mean = scores.mean()
        shrunk = (scores + self.tau * mean) / (1 + self.tau)
        shrunk = shrunk.clip(lower=0)
        if shrunk.sum() == 0:
            return EqualWeight().weight(selected)
        w = shrunk / shrunk.sum()
        return pd.DataFrame({"weight": w}, index=selected.index)


class AdaptiveBayesWeighting(BaseWeighting):
    """State-ful Bayesian weighting with exponential decay."""

    def __init__(
        self,
        *,
        half_life: int = 90,
        obs_sigma: float = 0.25,
        max_w: float | None = 0.20,
        prior_mean: str | FloatArray = "equal",
        prior_tau: float = 1.0,
    ) -> None:
        self.half_life = int(half_life)
        self.obs_tau = 1.0 / float(obs_sigma) ** 2
        self.max_w = None if max_w is None else float(max_w)
        self.prior_mean = prior_mean
        self.prior_tau = float(prior_tau)
        self.mean: pd.Series | None = None
        self.tau: pd.Series | None = None

    def _ensure_index(self, index: pd.Index) -> None:
        if self.mean is None:
            if isinstance(self.prior_mean, str) and self.prior_mean == "equal":
                m = np.repeat(1.0 / len(index), len(index))
            else:
                arr = np.asarray(self.prior_mean, dtype=float)
                if arr.shape[0] != len(index):
                    raise ValueError("prior_mean length mismatch")
                m = arr
            self.mean = pd.Series(m, index=index, dtype=float)
            self.tau = pd.Series(self.prior_tau, index=index, dtype=float)
        else:
            assert self.tau is not None
            for col in index:
                if col not in self.mean.index:
                    self.mean[col] = 1.0 / len(index)
                    self.tau[col] = self.prior_tau

    def update(self, scores: pd.Series, days: int) -> None:
        """Update posterior means and precisions with ``scores``."""

        self._ensure_index(scores.index)
        assert self.mean is not None and self.tau is not None  # for mypy

        if self.half_life > 0:
            decay = 0.5 ** (days / self.half_life)
            self.tau *= decay
        else:
            self.tau[:] = 0.0

        tau_old = self.tau.loc[scores.index]
        tau_new = tau_old + self.obs_tau
        m_old = self.mean.loc[scores.index]
        m_new = (m_old * tau_old + self.obs_tau * scores.astype(float)) / tau_new
        self.mean.loc[scores.index] = m_new
        self.tau.loc[scores.index] = tau_new

    def weight(self, candidates: pd.DataFrame, date: pd.Timestamp | None = None) -> pd.DataFrame:
        if len(candidates.index) == 0:
            return pd.DataFrame(columns=["weight"])
        self._ensure_index(candidates.index)
        assert self.mean is not None
        w = self.mean.reindex(candidates.index).fillna(0.0).clip(lower=0.0)
        if w.sum() == 0:
            w[:] = 1.0 / len(w)
        else:
            w /= w.sum()
        if self.max_w is not None:
            cap = self.max_w
            w = w.clip(upper=cap)
            total = w.sum()
            if total < 1.0:
                deficit = 1.0 - total
                room = w[w < cap]
                if not room.empty:
                    w.loc[room.index] += deficit / len(room)
                else:
                    w /= total
        return pd.DataFrame({"weight": w}, index=candidates.index)

    def get_state(self) -> dict[str, Any]:
        """Return a serialisable representation of the posterior state."""
        return {
            "mean": None if self.mean is None else self.mean.to_dict(),
            "tau": None if self.tau is None else self.tau.to_dict(),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Load posterior state from ``state``."""
        mean = state.get("mean")
        tau = state.get("tau")
        self.mean = None if mean is None else pd.Series(mean, dtype=float)
        self.tau = None if tau is None else pd.Series(tau, dtype=float)


__all__ = [
    "BaseWeighting",
    "EqualWeight",
    "ScorePropSimple",
    "ScorePropBayesian",
    "AdaptiveBayesWeighting",
]
